#pragma once
#include <limits>
#include <cstddef>

namespace viserion
{
	template<typename T>
	void normaliseVector(T* vec, size_t size)
	{
		T maxValue = std::numeric_limits<T>::min();

		for(size_t i = 0; i < size; ++i)
		{
			maxValue = (vec[i] > maxValue) ? vec[i] : maxValue;
		}

		for(size_t i = 0; i < size; ++i)
		{
			vec[i] /= maxValue;
		}
	}

}