#pragma once

#include <iostream>
#include <cassert>
#include <string>

class vector3d
{
public:
	double x;
	double y;
	double z;

	vector3d()
	{
		x = y = z = 0;
	}
	vector3d(const double newx, const double newy, const double newz)
	{
		x = newx;
		y = newy;
		z = newz;
	}
	vector3d(const vector3d& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	vector3d& operator=(const vector3d& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	vector3d operator-() const
	{
		return vector3d(-x, -y, -z);
	}

	vector3d operator+(const vector3d& v) const
	{
		return vector3d(x + v.x, y + v.y, z + v.z);
	}
	vector3d operator+=(const vector3d& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	vector3d operator-(const vector3d& v) const
	{
		return vector3d(x - v.x, y - v.y, z - v.z);
	}
	vector3d operator-=(const vector3d& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	vector3d operator*(const double scalar) const
	{
		return vector3d(scalar * x, scalar * y, scalar * z);
	}
	vector3d operator*=(const double scalar)
	{
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	vector3d operator/(const double scalar) const
	{
		return vector3d(x / scalar, y / scalar, z / scalar);
	}
	vector3d operator/=(const double scalar)
	{
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	//bool operator==(const vector3d& v) const
	//{
	//	return ((x == v.x) && (y == v.y) &&
	//		(z == v.z));
	//}
	//bool operator!=(const vector3d& v) const
	//{
	//	return !(*this == v);
	//}

	double& operator[](const unsigned int i)
	{
		switch (i)
		{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
		assert(0);
	}
	double operator[](const unsigned int i) const
	{
		switch (i)
		{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
		assert(0);
	}

	[[nodiscard]] double dot(const vector3d& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}
	[[nodiscard]] vector3d cross(const vector3d& v) const
	{
		return vector3d(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	[[nodiscard]] double norm() const
	{
		return sqrt(x * x + y * y + z * z);
	}
	[[nodiscard]] double normsquared() const
	{
		return x * x + y * y + z * z;
	}
	[[nodiscard]] vector3d normalized() const
	{
		return *this / this->norm();
	}

	[[nodiscard]] std::string toStr() const
	{
		return "[" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + "]";
	}

	void print() const
	{
		std::cout << "[" << x << ", " << y << ", " << z << "]";
	}

	[[nodiscard]] vector3d rot(char axis, double angle) const
	{
		double rotx[3][3] = { {1, 0, 0}, {0, cos(angle), -sin(angle)}, {0, sin(angle), cos(angle)} },
			roty[3][3] = { {cos(angle), 0, sin(angle)}, {0, 1, 0}, {-sin(angle), 0, cos(angle)} },
			rotz[3][3] = { {cos(angle), -sin(angle), 0}, {sin(angle), cos(angle), 0}, {0, 0, 1} };
		vector3d newVec;
		switch (axis)
		{
		case 'x':
			newVec[0] = rotx[0][0] * x + rotx[0][1] * y + rotx[0][2] * z;
			newVec[1] = rotx[1][0] * x + rotx[1][1] * y + rotx[1][2] * z;
			newVec[2] = rotx[2][0] * x + rotx[2][1] * y + rotx[2][2] * z;
			break;
		case 'y':
			newVec[0] = roty[0][0] * x + roty[0][1] * y + roty[0][2] * z;
			newVec[1] = roty[1][0] * x + roty[1][1] * y + roty[1][2] * z;
			newVec[2] = roty[2][0] * x + roty[2][1] * y + roty[2][2] * z;
			break;
		case 'z':
			newVec[0] = rotz[0][0] * x + rotz[0][1] * y + rotz[0][2] * z;
			newVec[1] = rotz[1][0] * x + rotz[1][1] * y + rotz[1][2] * z;
			newVec[2] = rotz[2][0] * x + rotz[2][1] * y + rotz[2][2] * z;
			break;
		default:
			std::cout << "Must choose x, y, or z rotation axis.";
			break;
		}
		return newVec;
	}
};

inline vector3d operator*(const double scalar, const vector3d& v)
{
	return v * scalar;
}

class rotation
{
public:
	rotation()
	{
		w = 1.0;
		x = y = z = 0;
	}
	rotation(const rotation& q)
	{
		w = q.w;
		x = q.x;
		y = q.y;
		z = q.z;
	}
	rotation(const double angle, const vector3d& axis)
	{
		w = cos(angle / 2.0);
		const vector3d goodaxis = axis.normalized();
		const double sinangle_over2 = sin(angle / 2.0);
		x = goodaxis.x * sinangle_over2;
		y = goodaxis.y * sinangle_over2;
		z = goodaxis.z * sinangle_over2;
	}

	rotation& operator=(const rotation& q)
	{
		w = q.w;
		x = q.x;
		y = q.y;
		z = q.z;
		return *this;
	}

	rotation operator*(const rotation& q) const
	{
		return rotation(w * q.w - x * q.x - y * q.y - z * q.z,
			w * q.x + x * q.w + y * q.z - z * q.y,
			w * q.y - x * q.z + y * q.w + z * q.x,
			w * q.z + x * q.y - y * q.x + z * q.w)
			.normalized();
	}
	rotation operator*=(const rotation& q)
	{
		*this = q * (*this);
		return *this;
	}

	bool operator==(const rotation& q) const
	{
		return ((w == q.w) && (x == q.x) &&
			(y == q.y) && (z == q.z));
	}
	bool operator!=(const rotation& q) const
	{
		return !(*this == q);
	}

	rotation conj() const
	{
		return rotation(w, -x, -y, -z);
	}

	vector3d rotate_vector(const vector3d& v) const
	{
		const rotation p(v.x * x + v.y * y + v.z * z,
			v.x * w - v.y * z + v.z * y,
			v.x * z + v.y * w - v.z * x,
			-v.x * y + v.y * x + v.z * w);
		const rotation product(w * p.w - x * p.x - y * p.y - z * p.z,
			w * p.x + x * p.w + y * p.z - z * p.y,
			w * p.y - x * p.z + y * p.w + z * p.x,
			w * p.z + x * p.y - y * p.x + z * p.w);
		return vector3d(product.x, product.y, product.z);
	}

	//void tostr(char str[]) const {
	//	const double theta = 2 * acos(w);
	//	const double fac = 1.0 / sin(theta / 2.0);
	//	sprintf(str, "[%6.2f, (%6.2f, %6.2f, %6.2f)]", theta, x*fac, y*fac, z*fac);
	//}

private:
	double w;
	double x;
	double y;
	double z;

	rotation(const double neww, const double newx, const double newy, const double newz)
	{
		w = neww;
		x = newx;
		y = newy;
		z = newz;
	}

	rotation operator/(const double scalar) const
	{
		return rotation(w / scalar, x / scalar, y / scalar, z / scalar);
	}
	rotation operator/=(const double scalar)
	{
		w /= scalar;
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	rotation normalized() const
	{
		return *this / sqrt(w * w + x * x + y * y + z * z);
	}
};

// Rounding
inline std::string rounder(double value, int digits)
{
	return std::to_string(value).substr(0, digits);
}

// Scientific Notation
inline std::string scientific(double value)
{
	std::stringstream ss;
	ss << value;
	std::string conversion = ss.str();
	return conversion;
}

// Output a nice title bar in terminal:
inline void titleBar(std::string title)
{
	std::cout << '\n';
	for (size_t i = 0; i < ((62 - title.size()) / 2); i++)
	{
		std::cout << '=';
	}
	std::cout << ' ' << title << ' ';
	for (size_t i = 0; i < ((62 - title.size()) / 2); i++)
	{
		std::cout << '=';
	}
	std::cout << "\n\n";
}

// // Print anything:
// template <typename theType>
// void print(theType value)
// {
// 	std::cout << value;
// }

// Ask a yes or no question:
inline bool input(const std::string& question)
{
	char answer;
	std::cout << question;
	std::cin >> answer;
	if (answer == 'y')
	{
		return true;
	}
	else
	{
		return false;
	}
}

// Generate a random double from -.5lim to .5lim so that numbers are distributed evenly around 0:
inline double randDouble(double lim)
{
	return lim * (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - .5);
}

// Returns a vector within the desired radius, resulting in spherical random distribution
inline vector3d randSphericalVec(double lim1, double lim2, double lim3)
{
	vector3d vec = { randDouble(lim1), randDouble(lim2), randDouble(lim3) };
	const double halfLim = lim1 * .5;
	while (vec.norm() > halfLim)
	{
		vec = { randDouble(lim1), randDouble(lim2), randDouble(lim3) };
	}
	return vec;
}

// Returns a vector within the desired radius, resulting in spherical random distribution
inline vector3d randShellVec(double lim, double innerRadius)
{
	vector3d vec = { randDouble(lim), randDouble(lim), randDouble(lim) };
	const double halfLim = lim * .5;
	if (halfLim < innerRadius)
	{
		std::cout << "Inner radius is larger than boundary. Impossible.\n";
		exit(EXIT_FAILURE);
	}
	while (vec.norm() > halfLim || vec.norm() < innerRadius)
	{
		vec = { randDouble(lim), randDouble(lim), randDouble(lim) };
	}
	return vec;
}

