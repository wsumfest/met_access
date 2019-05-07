class GeneralException():
	def __init__(self, exc):
		self.exc = exc

class NoDataFoundException():
	def __init__(self, id):
		self.id = id
