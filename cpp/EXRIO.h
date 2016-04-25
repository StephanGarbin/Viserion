#pragma once
#include <string>
#include <vector>

//OpenEXR
#include <ImfArray.h>
#include <ImathBox.h>
#include <ImfOutputFile.h>
#include <memory>

namespace viserion
{
	class EXRIO
	{
	public:

		EXRIO(const std::string& fileName);

		~EXRIO();

		int width();
		int height();
		int numChannels();

		//READING----------------------------------------
		bool readFloatToExistingPtr(float* data,
			int &width, int &height, int& numChannels,
			std::vector<std::string>& sortedChannelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			int multiPartIdx = -1);

		bool readFloatToNewPtr(float** data,
			int &width, int &height, int& numChannels,
			std::vector<std::string>& sortedChannelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			int multiPartIdx = -1);

		//WRITING----------------------------------------
		static bool writeFloat(const std::string& fileName,
			float* data,
			const std::vector<std::string>& channelNames,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow,
			Imf::Compression compression,
			int numParts = 0,
			int multiPartIdx = -1);

	private:
		std::shared_ptr<Imf::InputFile> m_inputFile;
	};

	bool readFloatEXR(const std::string& fileName,
			float** data,
			int &width, int &height, int& numChannels,
			Imath::Box2i& dataWindow,
			Imath::Box2i& displayWindow);
}



