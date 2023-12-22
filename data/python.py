import h5py
f = h5py.File('/home/lucas/Desktop/SpaceLab/jobs/test1/N_5/T_3/0_data.h5','r')

print(f['/energy'][:])
print(len(f['/energy'][:]))


# for item in f.keys():
#     print(item)
#     for it in f[item]:
#         print(it)
#         print (item + ":", f['/'+item])
        # print (item + ":", f['/'+item][it][:])