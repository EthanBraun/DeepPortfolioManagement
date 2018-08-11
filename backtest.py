import ccxt
from time import time
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize import *
from math import isnan
from random import random as rand 
from keras.models import *
from keras.layers import *
import keras.backend as K
from numpy.random import geometric
import tensorflow as tf

def expandDims(x):
	expX = K.expand_dims(x, axis=1)	
	expX = K.expand_dims(expX, axis=1)
	return expX


# Simulated crypto portfolio
class Portfolio():
	def __init__(self, symbols, weights, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs, noop=False, failureChance=0):
		self.symbols = symbols
		self.setWeights(weights)
		self.pBeta = pBeta
		self.k = k
		self.learningRate = learningRate
		self.minibatchCount = minibatchCount
		self.minibatchSize = minibatchSize
		self.epochs = epochs
		self.reset()
		self.tradeFee = 0.0005
		self.tradePer = 1.0 - self.tradeFee
		self.value = 1.0
		self.noop = noop
		self.failureChance = failureChance

	# Instantiate fully-convolutional ensemble of identical independent evaluators
	def createEiieNet(self, inputTensor, rates):
		biasIn = np.ones(1)
		mainInputShape = np.array(inputTensor).shape[1:]
		weightInputShape = np.array(rates).shape[1:]
		biasInputShape = biasIn.shape

		mIn = Input(shape=mainInputShape, name='mainInput')
		x = Conv2D(2, (3, 1))(mIn)
		x = Activation('relu')(x)
		x = Conv2D(20, (48, 1))(x)
		x = Activation('relu')(x)
		wIn = Input(shape=weightInputShape, name='weightInput') 
		wInExp = Lambda(expandDims)(wIn)
		x = Concatenate(axis=1)([x, wInExp])
		x = Conv2D(1, (1, 1))(x)
		bIn = Input(shape=biasInputShape, name='biasInput')
		bInExp = Lambda(expandDims)(bIn)
		x = Concatenate(axis=3)([x, bInExp])
		mOut = Activation('softmax')(x)
	
		model = Model([mIn, wIn, bIn], mOut) 
		model.compile(loss='mse', optimizer='nadam')
		self.model = model
		
		# Instantiate custom symbolic gradient
		#mu = K.placeholder(shape=(None, 1,))
		#y = K.placeholder(shape=(None, len(self.symbols),))
		mu = K.placeholder(shape=(1, ), name='mu')
		y = K.placeholder(shape=(len(self.symbols),), name='y')
		loss = -K.log(tf.multiply(mu, tf.tensordot(model.output, y, axes=1))) 
		grad = K.gradients(loss, model.trainable_weights)
		self.getGradient = K.function(inputs=[mIn, wIn, bIn, mu, y, model.output], outputs=grad) 
	
	def printParams(self):
		print('\nPortfolio parameters:')
		print('\tfailure chance: ' + str(self.failureChance))

	# Re-initialize portfolio state
	def reset(self):
		self.weights = [1.] + [0. for i in self.symbols[1:]]
		self.value = 1.0

	# Instantiate portfolio vector memory with initial values
	def initPvm(self, rates):
		#print('Initializing pvm')
		self.pvm = [[1.] + [0. for i in self.symbols[1:]] for j in (rates + rates[:1])]
		#print('PVM init shape: ' + str(np.array(self.pvm).shape))
		#print('pvm[0]: ' + str(self.pvm[0]))
		#print('pvm[100]: ' + str(self.pvm[100]))

	# Determine change in weights and portfolio value due to price movement between trading periods
	def updateRateShift(self, prevRates, curRates): 
		xHat = np.divide([1.] + curRates, [1.] + prevRates)
		values = [self.getValue() * w * x for w, x in zip(self.getWeights(), xHat)]

		prevValue = self.getValue()
		self.setValue(sum(values))

		b = np.divide(values, self.getValue())
		prevWeights = self.getWeights()
		
		self.setWeights(b)
		#return b, prevWeights, prevValue
		return b, prevWeights, self.getValue()

	# Sample the start index of a training minibatch from a geometric distribution
	def getMinibatchInterval(self, i):
		k = geometric(self.pBeta)
		tB = np.clip(-k - self.minibatchSize + i + 2, 1, i - self.minibatchSize + 1)
		return tB

	# Ascend reward gradient of minibatch starting at idx
	def trainOnMinibatch(self, idx, inTensor, rates):
		#print('pvm shape: ' + str(np.array(self.pvm).shape))
		#print('pvm[0]: ' + str(self.pvm[0]))
		#print('pvm[100]: ' + str(self.pvm[100]))

		pvmSeg = self.pvm[idx:idx + self.minibatchSize]
		#print('pvmSeg shape: ' + str(np.array(pvmSeg).shape))
		#print('pvmSeg: ' + str(pvmSeg))
		#print()

		truncPvmSeg = [q[1:] for q in pvmSeg]
		#print('truncPvmSeg shape: ' + str(np.array(truncPvmSeg).shape))
		#print('truncPvmSeg: ' + str(truncPvmSeg))
		#print()
		
		mIn = np.array(inTensor[idx:idx + self.minibatchSize])
		#wIn = np.array([w[1:] for w in self.pvm[idx:idx + self.minibatchSize]])
		wIn = np.array(truncPvmSeg)
		bIn = np.array([[1.0]] * self.minibatchSize)
	
		#print('wIn shape: ' + str(wIn.shape))
		#print('wIn: ' + str(wIn))
		#print()
		
		out = self.model.predict([mIn, wIn, bIn], batch_size=self.minibatchSize) 
		squeezeOut = np.squeeze(out)
		#print('\n\t\tout shape: ' + str(out.shape))

		pP = [[1.] + list(r) for r in rates[idx - 1:idx + self.minibatchSize - 1]]
		pC = [[1.] + list(r) for r in rates[idx:idx + self.minibatchSize]] 
		pN = [[1.] + list(r) for r in rates[idx + 1:idx + self.minibatchSize + 1]] 

		# Previous and current market relative price matrices
		yP = np.divide(pC, pP)
		yC = np.divide(pN, pC)	
		
		wPrev = np.array(self.pvm[idx:idx + self.minibatchSize])
		
		#print('yP shape: ' + str(yP.shape))
		#print('wPrev shape: ' + str(wPrev.shape))
		wpNum = np.multiply(yP, wPrev)
		wpDen = np.array([np.dot(ypT, wpT) for (ypT, wpT) in zip(yP, wPrev)])
		#print('\n\nwpNum shape: ' + str(wpNum.shape))
		#print('wpDen shape: ' + str(wpDen.shape))
		#print('\n')
		wPrime = [np.divide(n, d) for (n, d) in zip(wpNum, wpDen)]

		#print('len squeeze out: ' + str(len(squeezeOut)))

		mu = [[self.calculateMu(wPT, wT, self.k)] for (wPT, wT) in zip(wPrime, squeezeOut)]
		
		#print('mIn shape: ' + str(mIn.shape))
		#print('wIn shape: ' + str(wIn.shape))
		#print('bIn shape: ' + str(bIn.shape))
		#print('mu shape: ' + str(np.array(mu).shape))
		#print('yC shape: ' + str(yC.shape))
		#print('out shape: ' + str(out.shape))

		#grad = [self.getGradient(inputs=[[mT], [wT], [bT], [muT], [yT], [oT]]) for (mT, wT, bT, muT, yT, oT) in zip(mIn, wIn, bIn, mu, yC, out)]  
		grad = [self.getGradient(inputs=[[mT], [wT], [bT], muT, yT, [oT]]) for (mT, wT, bT, muT, yT, oT) in zip(mIn, wIn, bIn, mu, yC, out)]  
		batchGrad = np.sum(grad, axis=0)

		#print('\n\n\t\t\tAFTER GET GRADIENT\n\n')
		updates = [self.learningRate * g for g in batchGrad]
		
		modelWeights = self.model.get_weights()
		updateWeights = [np.add(wT, uT) for (wT, uT) in zip(modelWeights, updates)]
		"""
		print()
		print()
		print('updates shape: ' + str(np.array(updates).shape))
		print('modelWeights shape: ' + str(np.array(modelWeights).shape))
		print()
		print()
		"""
		self.model.set_weights(updateWeights)

	# RL agent training function
	def train(self, inTensor, rates):
		self.initPvm(rates)
		# For each epoch
		for epoch in range(self.epochs):
			print('\nBeginning epoch ' + str(epoch))
			self.reset()
			# For each trading period in the interval
			for i, (r, p, x) in enumerate(zip(rates[1:], self.pvm[1:-1], inTensor[1:])):
				# Determine eiie output at the current period
				mIn = np.array([x])
				wIn = np.array([np.squeeze(p[1:])])
				bIn = np.array([1.])
				modelOutput = self.model.predict([mIn, wIn, bIn])[0]	
				#print('modelOutput[0] shape: ' + str(modelOutput.shape))

				# Overwrite pvm at subsequent period
				self.pvm[i + 2] = modelOutput[0][0]
				
				# Update portfolio for current timestep
				#newB, prevB, prevValue = self.updateRateShift(rates[i], r) 
				#self.updatePortfolio(newB, prevB, prevValue, rates[i], r) 
				newB, prevB, curValue = self.updateRateShift(rates[i], r) 
				self.updatePortfolio(modelOutput[0][0], newB, curValue, rates[i], r) 
				if i % 1000 == 0:
					print('\ti (' + str(i) + ') value: ' + str(self.getValue()))
				 
				# Train EIIE over minibatches of historical data
				if i - self.minibatchSize >= 0:
					for j in range(self.minibatchCount):
						# Sample minibatch interval from geometric distribution
						idx = self.getMinibatchInterval(i)
						self.trainOnMinibatch(idx, inTensor, rates)
			print('Epoch ' + str(epoch) + ' value: ' + str(self.getValue()))
	
	# Calculate current portfolio value and set portfolio weights
	def updatePortfolio(self, newWeights, prevWeights, prevValue, prevRates, curRates):
		# Calculate current portfolio value
		rateRatios = [1.] + list(np.divide(curRates, prevRates))
		prevValues = np.multiply(prevWeights, prevValue)
		currentValues = np.multiply(rateRatios, prevValues)
		currentVal = sum(currentValues)
		
		# Calculate difference between current and new weights
		weightDelta = np.subtract(newWeights, self.weights)

		valueDelta = [(self.value * delta) for delta in weightDelta]

		# Calculate BTC being bought
		buy = self.tradePer * -sum([v if (v < 0) else 0 for v in valueDelta])

		posValDeltas = {}
		for i, v in enumerate(valueDelta):
			if v > 0:
				posValDeltas[i] = v

		posValDeltaSum = sum(posValDeltas.values())
		posValDeltaPer = np.divide(list(posValDeltas.values()), posValDeltaSum)

		# Calculate actual positive value changes with trade fees
		realPosValDeltas = [per * self.tradePer * buy for per in posValDeltaPer]

		# Calculate overall value deltas
		realValDeltas = []
		for val in valueDelta:
			if val <= 0:
				realValDeltas.append(val)
			else:
				realValDeltas.append(realPosValDeltas.pop(0))

		# Simulate possiblility of trade failure
		for i in range(1, len(realValDeltas)):
			if rand() < self.failureChance:
				realValDeltas[0] += realValDeltas[i] / self.tradePer	
				realValDeltas[i] = 0
		

		# Calculate new value
		newValues = np.add(currentValues, realValDeltas)
		newValue = sum(newValues)
		self.setValue(newValue)
	
		self.setWeights(np.divide(newValues, newValue))


	# Iteratively calculate the transaction remainder factor for the period
	def calculateMu(self, wPrime, w, k):
		"""
		print('wPrime type: ' + str(type(wPrime)))
		print('w type: ' + str(type(w)))
		print()

		print('wPrime shape: ' + str(np.array(wPrime).shape))
		print('w shape: ' + str(np.array(w).shape))
		"""

		# Calculate initial mu value
		mu = self.tradeFee * sum([abs(wpI - wI) for wpI, wI in zip(wPrime, w)]) 	

		# Calculate iteration of mu
		for i in range(k):
			#print('pre muSuff: ' + str([(wpI - mu * wI) for (wpI, wI) in zip(wPrime, w)]))
			muSuffix = sum([max((wpI - mu * wI), 0) for (wpI, wI) in zip(wPrime, w)])
			mu = (1. / (1. - self.tradeFee * w[0])) * (1. - (self.tradeFee * wPrime[0]) - (2 * self.tradeFee - (self.tradeFee ** 2)) * muSuffix)
		return mu


	# Simulate the pamr agent over a set of data and return the final portfolio value
	def simulate(self, fData):	
		x = [1. for i in symbols]

		for i, _ in enumerate(fData[:-1]):
			# Get market-relative price vector at current timestep
			prevRates = []
			curRates = []
			xHat = []
			values = []

			for j, _ in enumerate(symbols):
				lastClose = fData[i+1][j]
				prevClose = fData[i][j]
				curRates.append(lastClose)
				prevRates.append(prevClose)
				xHat.append(lastClose / prevClose)
				values.append(self.getWeights()[j] * self.getValue() * xHat[j])
	
			# Update overall market-relative price vector over interval
			x = np.multiply(x, xHat)

			prevValue = self.getValue()
			self.setValue(sum(values))
			self.values.append(self.getValue())

			b = np.divide(values, self.getValue())
			prevWeights = self.getWeights()
			
			self.setWeights(b)
		
			# Redistribute portfolio on an interval
			if (not self.noop) and (i % self.interval == 0):
				# Calculate loss
				loss = max(0, np.dot(b, x) - self.getEpsilon())

				# Calculate Tau (update step size)
				tau = loss / ((norm(np.subtract(x, np.mean(x))) ** 2) + (1 / (2. * self.getSlack()))) 

				# Calculate new portfolio weights
				b = np.subtract(self.getWeights(), np.multiply(tau, np.subtract(x, np.mean(x))))
			
				# Project portfolio into simplex domain
				result = minimize(lambda q: norm(np.subtract(q, b)) ** 2, [1. / len(b) for z in b], method='SLSQP', bounds=[(0.0, 1.0) for z in b], constraints={'type': 'eq', 'fun': lambda q: sum(q) - 1.0})
			
				# Update portfolio with projected new weights
				self.updatePortfolio(result['x'], prevWeights, prevValue, prevRates, curRates) 

				# Reset overall market-relative price vector
				x = [1. for i in symbols]
		print('\n\tFinal Weights: ' + str(np.array(self.getWeights())) + '\n') 
		return self.getValue()

	def getLabel(self, name):
		return (name + ' - (epsilon: ' + str(self.epsilon) + ', slack: ' + str(self.slack) + ', interval: ' + str(self.interval) + ', failure chance: ' + str(self.failureChance) + ')')  

	def getEpsilon(self):
		return self.epsilon

	def getSlack(self):
		return self.slack

	def getValue(self):
		return self.value

	def getWeights(self):
		return self.weights[:]

	def getValues(self):
		return self.values

	def setValue(self, value):
		self.value = value

	# Assign new portfolio weights
	def setWeights(self, weights):
		self.weights = weights[:]


def getData(b, t, depth, symbol):
	print('\nGetting data for symbol: ' + symbol)
	data = []
	while True:
		ohlcv = b.fetch_ohlcv(symbol, since=t)
		data = ohlcv[:] if len(data) == 0 else np.concatenate((ohlcv, data))
		t -= 500 * 60000
		if len(data) >= depth:
			break
	return data

# Make sure there are no gaps or repeats in a sequence of timesteps
def validateTimesteps(data):
	print('Final data len: ' + str(len(data)))
	timesteps = [x[0] for x in data]
	for i, _ in enumerate(timesteps[:-1]):
		if timesteps[i + 1] - timesteps[i] != 60000:
			#plt.plot(timesteps, color='r')
			#plt.show()
			return False
	return True

# Trim the starts of sequences to ensure similar length for all symbols
def truncateData(data):
	truncLength = min([len(sym) for sym in data])
	print('Truncated data len: ' + str(truncLength))
	#return [x[:truncLength] for x in data]
	return [x[len(x) - truncLength:] for x in data]

# Ensures the start of each sequence is the same after being truncated
def checkTruncData(data):
	timestamp = data[0][0][0]
	for i in range(1, len(data)):
		if data[i][0][0] != timestamp:
			print('\t\t\tTimestamps not synchronized!')
			return
	print('\t\t\tTimestamps synchronized')

# Reformat the data into a single multivariate sequence
def formatData(data, addBtc):
	# (symbols x timesteps x features) --> (timesteps x features x symbols)
	features = [3, 2, 4]
	fData = []
	print('pre-formatted data shape: ' + str(np.array(data).shape))
	for i, _ in enumerate(data[0]):
		stepData = []
		for idx in features:
			try:
				featVec = [data[j][i][idx] for j, _ in enumerate(data)]
				stepData.append(np.insert(featVec, 0, 1.) if addBtc else featVec)
			except Exception:
				print('Exception occured with (i, idx) = (' + str(i) + ', ' + str(idx) + ')')
				raise
		fData.append(stepData)
	return fData

# Reformat the data into a sequence of normalized input tensors (timesteps x features x lookback x symbols)
# 	also return y (the list of market-relative price vectors)
def formatDataForInput(data, window):
	x = []
	y = []
	rates = []
	for i in range(window, len(data)):
		stepData = []
		for j, _ in enumerate(data[i]):
			stepData.append([np.divide(data[k][j], data[i - 1][2])  for k in range(i - window, i)])
		x.append(stepData)
		y.append(np.divide(data[i - 1][2], data[i - 2][2]))
		rates.append(data[i][2])
	return x, y, rates

now = int(time() * 1000)
start = now - 500 * 60000
binance = ccxt.binance()
binance.load_markets()
#symbols = ['DENT/BTC', 'ETH/BTC', 'ETC/BTC', 'EOS/BTC', 'MFT/BTC', 'KEY/BTC', 'NPXS/BTC', 'NEO/BTC', 'ICX/BTC', 'QKC/BTC', 'XRP/BTC', 'LOOM/BTC', 'ONT/BTC', 'ADA/BTC']

symbols = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'ICX/BTC', 'XRP/BTC', 'XLM/BTC', 'NEO/BTC', 'LTC/BTC', 'ADA/BTC']

#symbols = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'XRP/BTC', 'NEO/BTC','ADA/BTC']

#symbols = ['EOS/BTC', 'ETH/BTC']

#symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['TRX/BTC', 'ETC/BTC', 'BCH/BTC', 'IOTA/BTC', 'ZRX/BTC', 'WAN/BTC', 'WAVES/BTC', 'SNT/BTC', 'MCO/BTC', 'DASH/BTC', 'ELF/BTC', 'AION/BTC', 'STRAT/BTC', 'XVG/BTC', 'EDO/BTC', 'IOST/BTC', 'WABI/BTC', 'SUB/BTC', 'OMG/BTC', 'WTC/BTC', 'LSK/BTC', 'ZEC/BTC', 'STEEM/BTC', 'QSP/BTC', 'SALT/BTC', 'ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC']
#depth = 110000

#depth = 210000
depth = 110000
#clip = 35000
clip = 99000
holdBtc = True
window = 50

print('\nPortfolio symbols: ' + str(symbols))
print('Managing ' + str(len(symbols)) + ' alts in each portfolio')
print('Holding BTC: ' + str(holdBtc))

data = []
for sym in symbols:
	symData = getData(binance, start, depth, sym)
	symData = symData[:-clip]
	valid = validateTimesteps(symData)
	if valid:
		print('Data valid for ' + sym)
	else:
		print('Data invalid for ' + sym + '!!!')
	data.append(symData)
	
tData = truncateData(data)
checkTruncData(tData)
fData = formatData(tData, False)
x, y, rates = formatDataForInput(fData, window)


# Modify symbols and data if portfolios can hold BTC
if holdBtc:
	symbols.insert(0, 'BTC/BTC')

print('\n\nx shape: ' + str(np.array(x).shape))
print('y shape: ' + str(np.array(y).shape))
b = [1.] + [0.] * (len(symbols) - 1)  
pBeta = 0.00005
k = 15
learningRate = 0.00003
minibatchCount = 30
minibatchSize = 50
epochs = 50

port = Portfolio(symbols, b, pBeta, k, learningRate, minibatchCount, minibatchSize, epochs)
port.createEiieNet(x, y)
port.train(x, rates)

print('\n' + str(port.model.summary()))


"""
# Initialize simulated portfolio
port0 = Portfolio(symbols, 0.25, 7, 15, b)
port1 = Portfolio(symbols, 0.35, 7, 15, b)
port2 = Portfolio(symbols, 0.35, 8, 30, b, failureChance=0.025)
port3 = Portfolio(symbols, 0.40, 9, 30, b)
port4 = Portfolio(symbols, 0.25, 9, 30, b)
port5 = Portfolio(symbols, 0.35, 9, 30, b)
port6 = Portfolio(symbols, 0.35, 9, 30, b, failureChance=0.05)
port7 = Portfolio(symbols, 0.35, 9, 30, b, failureChance=0.10)

ports = [port0, port1, port2, port3, port4, port5, port6, port7]
bh = Portfolio(symbols, 0.95, 3, 1, b, noop=True)

print('\nBuy & Hold:')
val = bh.simulate(fData)
print('\tPortfolio value: ' + str(val) + '\n')

for port in ports:
	port.printParams()
	print('---Simulating portfolio---')
	val = port.simulate(fData)
	print('\tPortfolio value: ' + str(val) + '\n')


names = ['Portfolio ' + str(i) for i in range(len(ports))]
labels = [ports[i].getLabel(names[i]) for i in range(len(ports))]
colors = ['#FF0000', '#FF9000', '#FFFF00', '#00FF00', '#00D8FF', '#0000FF', '#9800FF', '#FA00FF']

plt.title('Portfolio value v. Time')
plt.xlabel('Minutes')
plt.ylabel('BTC')
plt.plot(bh.getValues(), label='Buy & Hold', color='#000000')
for i in range(len(ports)):
	plt.plot(ports[i].getValues(), label=labels[i], color=colors[i])
plt.legend()
plt.show()
"""
