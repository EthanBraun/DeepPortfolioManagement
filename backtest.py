import ccxt
from time import time
import numpy as np
from matplotlib import pyplot as plt
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
		mu = K.placeholder(shape=(None, 1), name='mu')
		y = K.placeholder(shape=(None, len(self.symbols)), name='y')

		sqOut = K.squeeze(K.squeeze(model.output, 1), 1)

		yOutMult = tf.multiply(sqOut, y)
		yOutBatchDot = tf.reduce_sum(yOutMult, axis=1, keep_dims=True)
		muDotMult = tf.multiply(mu, yOutBatchDot)		

		loss = -K.log(muDotMult)

		grad = K.gradients(loss, model.trainable_weights)
		self.getGradient = K.function(inputs=[mIn, wIn, bIn, mu, y, model.output], outputs=grad) 

		print('mIn shape: ' + str(mIn.get_shape().as_list()))
		print('wIn shape: ' + str(wIn.get_shape().as_list()))
		print('bIn shape: ' + str(bIn.get_shape().as_list()))
		print('mu shape: ' + str(mu.get_shape().as_list()))
		print('y shape: ' + str(y.get_shape().as_list()))
		print('sqOut shape: ' + str(sqOut.get_shape().as_list()))
		print('\nyOutMult shape: ' + str(yOutMult.get_shape().as_list()))
		print('yOutBatchDot shape: ' + str(yOutBatchDot.get_shape().as_list()))
		print('muDotMult shape: ' + str(muDotMult.get_shape().as_list()))
		print('\nloss shape: ' + str(loss.get_shape().as_list()))
		print('grad shape: ' + str([g.get_shape().as_list() for g in grad]))
	
	def printParams(self):
		print('\nPortfolio parameters:')
		print('\tfailure chance: ' + str(self.failureChance))

	# Re-initialize portfolio state
	def reset(self):
		self.weights = [1.] + [0. for i in self.symbols[1:]]
		self.value = 1.0

	# Instantiate portfolio vector memory with initial values
	def initPvm(self, rates):
		self.pvm = [[1.] + [0. for i in self.symbols[1:]] for j in (rates + rates[:1])]

	# Determine change in weights and portfolio value due to price movement between trading periods
	def updateRateShift(self, prevRates, curRates): 
		xHat = np.divide([1.] + curRates, [1.] + prevRates)
		values = [self.getValue() * w * x for w, x in zip(self.getWeights(), xHat)]

		prevValue = self.getValue()
		self.setValue(sum(values))

		b = np.divide(values, self.getValue())
		prevWeights = self.getWeights()
		
		self.setWeights(b)
		return b, prevWeights, self.getValue()

	# Sample the start index of a training minibatch from a geometric distribution
	def getMinibatchInterval(self, i):
		k = geometric(self.pBeta)
		tB = np.clip(-k - self.minibatchSize + i + 2, 1, i - self.minibatchSize + 1)
		return tB

	# Ascend reward gradient of minibatch starting at idx
	def trainOnMinibatch(self, idx, inTensor, rates):
		pvmSeg = self.pvm[idx:idx + self.minibatchSize]

		truncPvmSeg = [q[1:] for q in pvmSeg]
		
		mIn = np.array(inTensor[idx:idx + self.minibatchSize])
		wIn = np.array(truncPvmSeg)
		bIn = np.array([[1.0]] * self.minibatchSize)
	
		out = self.model.predict([mIn, wIn, bIn], batch_size=self.minibatchSize) 
		squeezeOut = np.squeeze(out)

		pP = [[1.] + list(r) for r in rates[idx - 1:idx + self.minibatchSize - 1]]
		pC = [[1.] + list(r) for r in rates[idx:idx + self.minibatchSize]] 
		pN = [[1.] + list(r) for r in rates[idx + 1:idx + self.minibatchSize + 1]] 

		# Previous and current market relative price matrices
		yP = np.divide(pC, pP)
		yC = np.divide(pN, pC)	
		
		wPrev = np.array(self.pvm[idx:idx + self.minibatchSize])
		
		wpNum = np.multiply(yP, wPrev)
		wpDen = np.array([np.dot(ypT, wpT) for (ypT, wpT) in zip(yP, wPrev)])
		wPrime = [np.divide(n, d) for (n, d) in zip(wpNum, wpDen)]

		mu = [[self.calculateMu(wPT, wT, self.k)] for (wPT, wT) in zip(wPrime, squeezeOut)]
		
		grad = self.getGradient(inputs=[mIn, wIn, bIn, mu, yC, out])  

		updates = [self.learningRate * g for g in grad]
		
		modelWeights = self.model.get_weights()
		updateWeights = [np.add(wT, uT) for (wT, uT) in zip(modelWeights, updates)]
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

				# Overwrite pvm at subsequent period
				self.pvm[i + 2] = modelOutput[0][0]
				
				# Update portfolio for current timestep
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
		# Calculate initial mu value
		mu = self.tradeFee * sum([abs(wpI - wI) for wpI, wI in zip(wPrime, w)]) 	

		# Calculate iteration of mu
		for i in range(k):
			muSuffix = sum([max((wpI - mu * wI), 0) for (wpI, wI) in zip(wPrime, w)])
			mu = (1. / (1. - self.tradeFee * w[0])) * (1. - (self.tradeFee * wPrime[0]) - (2 * self.tradeFee - (self.tradeFee ** 2)) * muSuffix)
		return mu

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
# 	(Often the api calls return repeating segments at the end of a sequence)
def truncateData(data):
	truncLength = min([len(sym) for sym in data])
	print('Truncated data len: ' + str(truncLength))
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

symbols = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'ICX/BTC', 'XRP/BTC', 'XLM/BTC', 'NEO/BTC', 'LTC/BTC', 'ADA/BTC']

depth = 15000
clip = 5000
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

# Modify symbols if portfolios can hold BTC
if holdBtc:
	symbols.insert(0, 'BTC/BTC')

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
