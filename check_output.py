

def main():
	path = "/home/lpkolanz/Desktop/SpaceLab/jobs/singleCoreComparison1/"
	path = "/home/lpkolanz/Desktop/SpaceLab/jobs/multiCoreTest2/"
	file = "25_2_R4e-05_v4e-01_cor0.63_mu0.1_rho2.25_k4e+00_Ha5e-12_dt5e-10_simData.csv"

	print("Line, elements, balls")
	count = 1
	countline = 1
	with open(path+file) as fp:
		while 1:
			char = fp.read(1)
			if not char:
				break
			elif char == ',':
				count+=1
			elif char == '\n':
				print("{}, {}, {}".format(countline,count,count/11))
				count = 1
				countline += 1


if __name__ == '__main__':
	main()