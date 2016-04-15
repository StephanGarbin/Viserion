local ffi = require 'ffi'

ffi.cdef[[
	struct ViserionImageIO;
	typedef struct ViserionImageIO ViserionImageIO_t;
	ViserionImageIO_t* createIOInstance(int);
	void destroyIOInstance(ViserionImageIO_t*);
	bool getNextImage(ViserionImageIO_t*);

	bool readEXR(ViserionImageIO_t*, const char*, THFloatTensor*);

	bool reset(ViserionImageIO_t*);

	bool createDataSetCache(const char*, const char*, const char*, bool, bool, bool);
]]

clib = ffi.load('/home/stephan/luaProjects/Viserion/ViserionImageIO')

