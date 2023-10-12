import numpy as np

def main():
	with open("test_loop_decrease_order_precise.out") as textfile1, open("test_loop_increase_order_precise.out") as textfile2: 
		for i in range(3000000):
			line1 = textfile1.readline()
			line2 = textfile2.readline()

			temp1 = line1.split(" ")[:-1]
			temp2 = line2.split(" ")[:-1]

			# print(temp1)
			# print(temp2)


			for j in range(3):
				try:
					if temp1[j] != temp2[j]:
						print(i)
						print(temp1[0]+", "+temp1[1]+", "+temp1[2]+" :: "+temp2[0]+", "+temp2[1]+", "+temp2[2])
						exit(0)
				except:
					print(i)
					print(temp1)
					print(temp2)
					exit(0)

			# temp  = np.array([-10,0,0])

			# for j in range(3):
			# 	temp11 = np.array([float(e) for e in temp1[j].split(',')])
			# 	temp22 = np.array([float(e) for e in temp2[j].split(',')])

			# 	temp[j]   = np.linalg.norm(temp22 - temp11)

			# if (temp[0] != 0) or (temp[1] != 0) or (temp[2] != 0):
			# 	print(str(i) + " not good")
			# 	exit()

			# else:
			# 	print(i)


if __name__ == '__main__':
	main()