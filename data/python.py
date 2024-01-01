import h5py
f = h5py.File('/home/lucas/Desktop/SpaceLab/jobs/restartTest3/N_10/T_3/2_data.h5','r')

print(f['/constants'][:])
print(len(f['/constants'][:]))


# for item in f.keys():
#     print(item)
#     for it in f[item]:
#         print(it)
#         print (item + ":", f['/'+item])
        # print (item + ":", f['/'+item][it][:])