#include "scratchPad.h"

int main()
{
	vector3d pos(1,0, 0), vel(-.1,1, 0);
	cout << acos(vel.normalized().dot(pos.normalized()))*180/3.14159;
}