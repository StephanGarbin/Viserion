#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "TH/THTensor.h"

class ViserionImageIO
{
public:
	ViserionImageIO(int val);
	~ViserionImageIO();

	bool getNextImage();

	bool reset();

	bool readEXR(const std::string& fileName, float** destination, long& numRows, long& numCols, long& numChannels);

private:
	int m_val;
};

bool createDataSetCache(const std::string& datasetName, const std::string& directory,
	const std::string& cacheDirectory, const std::string& compression, bool recursive, bool normalise, bool whiten);

void loadPNG(const std::string& fileName, std::vector<float>& pixels, int& width, int& height);


extern "C" ViserionImageIO* createIOInstance(int val)
{
	return new ViserionImageIO(val);
}

extern "C" void destroyIOInstance(ViserionImageIO* ptr)
{
	delete ptr;
}

extern "C" bool getNextImage(ViserionImageIO* ptr)
{
	return ptr->getNextImage();
}

extern "C" bool reset(ViserionImageIO* ptr)
{
	return ptr->reset();
}

extern "C" bool createDataSetCache(const char* datasetName, const char* directory, const char* cacheDirectory, const* char compression, bool recursive, bool normalise, bool whiten)
{
	return createDataSetCache(std::string(datasetName), std::string(directory), std::string(cacheDirectory), std::string(compression), recursive, normalise, whiten);
}

extern "C" bool readEXR(ViserionImageIO* ptr, const char *fileName, THFloatTensor* targetTensor)
{
	float* data;

	long numRows;
	long numCols;
	long numChannels;

	bool status = ptr->readEXR(std::string(fileName), &data, numRows, numCols, numChannels);

	if(status == false)
	{
		return false;
	}

	//std::cout << numRows << ", " << numCols << ", " << numChannels << std::endl;

	THFloatStorage* tensorData = THFloatStorage_newWithData(data, numRows * numCols * numChannels);

	long sizeStorage[3]   = { numChannels, numCols, numRows};
    long strideStorage[3] = { numRows * numCols, numRows, 1 }; //contiguous last dimension
        
    THLongStorage* size    = THLongStorage_newWithData(sizeStorage, 3);
    THLongStorage* stride  = THLongStorage_newWithData(strideStorage, 3);
	
	THFloatStorage_free(targetTensor->storage);        
    THFloatTensor_setStorage(targetTensor, tensorData, 0LL, size, stride);

    delete[] data; //--lua takes ownership of ptr??

    return true;
}
