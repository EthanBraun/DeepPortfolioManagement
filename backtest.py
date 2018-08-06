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


def expandDims(x):
	expX = K.expand_dims(x, axis=1)	
	expX = K.expand_dims(expX, axis=1)
	return expX


# Simulated crypto portfolio
class Portfolio():
	def __init__(self, symbols, epsilon, slack, interval, weights, noop=False, failureChance=0):
		self.symbols = symbols
		self.epsilon = epsilon
		self.slack = slack
		self.interval = interval
		self.setWeights(weights)
		self.reset()
		self.tradeFee = 0.0005
		self.tradePer = 1.0 - self.tradeFee
		self.value = 1.0
		self.noop = noop
		self.failureChance = failureChance

	# Instantiate fully-convolutional ensemble of identical independent evaluators
	def createEiieNet(self, inputTensor, mrpVector):
		biasIn = np.ones(1)
		mainInputShape = np.array(inputTensor).shape[1:]
		weightInputShape = np.array(mrpVector).shape[1:]
		biasInputShape = biasIn.shape
		print('mainInputShape: ' + str(mainInputShape))
		print('weightInputShape: ' + str(weightInputShape))
		print('biasInputShape: ' + str(biasInputShape))
		
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
	
	def printParams(self):
		print('\nPortfolio parameters:')
		print('\tepsilon: ' + str(self.epsilon))
		print('\tslack: ' + str(self.slack))
		print('\tinterval: ' + str(self.interval))
		print('\tfailure chance: ' + str(self.failureChance))

	# Re-initialize portfolio state
	def reset(self):
		self.weights = [1.] + [0. for i in self.symbols]
		self.value = 1.0

	# Instantiate portfolio vector memory with initial values
	def initPvm(self, rates):
		self.pvm = [[1.] + [0. for i in self.symbols] for j in (rates + rates[:1])]

	# Determine change in weights and portfolio value due to price movement between trading periods
	def updateRateShift(self, prevRates, curRates): 
		xHat = np.divide([1.] + curRates, [1.] + prevRates)
		values = [self.getValue() * w * x for w, x in zip(self.getWeights(), xHat)]

		prevValue = self.getValue()
		self.setValue(sum(values))

		b = np.divide(values, self.getValue())
		prevWeights = self.getWeights()
		
		self.setWeights(b)
		return prevWeights, b

	# RL agent training function
	def train(self, inTensor, rates):
		self.initPvm(rates)
		# For each epoch
		for epoch in range(self.epochs):
			self.reset()
			# For each trading period in the interval
			for i, (r, p, x) in enumerate(zip(rates, self.pvm[:-1], inTensor)):
				# Determine eiie output at the current period
				modelInput = np.array([[x, p[1:], [1.]]]) 
				modelOutput = self.model.predict(modelInput)[0]	
				# Overwrite pvm at subsequent period
				self.pvm[i + 1] = modelOutput
				  
	
	# Calculate current portfolio value and set portfolio weights
	def updatePortfolio(self, newWeights, prevWeights, prevValue, prevRates, curRates):
		# Calculate current portfolio value
		rateRatios = np.divide(curRates, prevRates)
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
			muSuffix = sum([max((wpI - mu * wI), 0) for wpI, wI in zip(wPrime, w)])
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
			plt.plot(timesteps, color='r')
			plt.show()
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
	for i in range(window, len(data)):
		stepData = []
		for j, _ in enumerate(data[i]):
			stepData.append([np.divide(data[k][j], data[i - 1][2])  for k in range(i - window, i)])
		x.append(stepData)
		y.append(np.divide(data[i - 1][2], data[i - 2][2]))
	return x, y

now = int(time() * 1000)
start = now - 500 * 60000
binance = ccxt.binance()
binance.load_markets()
#symbols = ['DENT/BTC', 'ETH/BTC', 'ETC/BTC', 'EOS/BTC', 'MFT/BTC', 'KEY/BTC', 'NPXS/BTC', 'NEO/BTC', 'ICX/BTC', 'QKC/BTC', 'XRP/BTC', 'LOOM/BTC', 'ONT/BTC', 'ADA/BTC']

#symbols = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'ICX/BTC', 'XRP/BTC', 'XLM/BTC', 'NEO/BTC', 'LTC/BTC', 'ADA/BTC']
symbols = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'XRP/BTC', 'NEO/BTC','ADA/BTC']
#symbols = ['EOS/BTC', 'ETH/BTC']

#symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['TRX/BTC', 'ETC/BTC', 'BCH/BTC', 'IOTA/BTC', 'ZRX/BTC', 'WAN/BTC', 'WAVES/BTC', 'SNT/BTC', 'MCO/BTC', 'DASH/BTC', 'ELF/BTC', 'AION/BTC', 'STRAT/BTC', 'XVG/BTC', 'EDO/BTC', 'IOST/BTC', 'WABI/BTC', 'SUB/BTC', 'OMG/BTC', 'WTC/BTC', 'LSK/BTC', 'ZEC/BTC', 'STEEM/BTC', 'QSP/BTC', 'SALT/BTC', 'ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC', 'VEN/BTC', 'NAV/BTC', 'BQX/BTC']
#symbols = ['ETH/BTC', 'XRP/BTC', 'XLM/BTC', 'ADA/BTC', 'NEO/BTC', 'XMR/BTC', 'XEM/BTC', 'EOS/BTC', 'ICX/BTC', 'LTC/BTC', 'QTUM/BTC']
#depth = 110000

#depth = 210000
depth = 110000
#clip = 35000
clip = 60000
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
x, y = formatDataForInput(fData, window)


# Modify symbols and data if portfolios can hold BTC
if holdBtc:
	symbols.insert(0, 'BTC/BTC')

print('\n\nx shape: ' + str(np.array(x).shape))
print('y shape: ' + str(np.array(y).shape))
b = [1 / float(len(symbols))] * len(symbols)

port = Portfolio(symbols, 0.25, 9, 30, b)
port.createEiieNet(x, y)

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
