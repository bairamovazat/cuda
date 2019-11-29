#pragma once

#include <stdexcept>

template <typename T>
class Vector
{
public:
	T* pointer;
	size_t size;

	Vector(size_t size) {
		this->size = size;
		this->pointer = new T[size];
	}

	~Vector() {
		delete[] this->getPointer();
	}

	T* getPointer() {
		T* Vector<T>::getPointer();
		return this->pointer;
	}

	T getElement(size_t index) {
		return *(this->pointer + index);
	}

	void setElement(size_t index, T value) {
		if (this->getSize() < index) {
			throw std::out_of_range("Out of range");
		}
		return *(this->pointer + index) = value;
	}

	size_t getSize() {
		return this->size;
	}

	Vector<T> plus(Vector<T> rv) {
		size_t elementCount = (rv.getSize() > this->getSize() ? rv.getSize() : this->getSize());
		Vector<T> result = new Vector(elementCount);
		for (size_t i = 0; i < elementCount; i++) {
			result.setElement(i, ((i < this->getSize() ? this->getElement(i) : 0) +
				(i < rv.getSize() ? rv.getElement(i) : 0)
				)
			);
		}
		return result;
	}

private:

};
