#include "H5Cpp.h"
#include <filesystem>
#include <iostream>
#include <vector>

class HDF5Handler {
    public:
        HDF5Handler(std::string filename,ssize_t max_size=-1) : filename(filename),max_size(max_size) 
        {
        	if (max_size > 0)
        	{
        		fixed = true;
        	}
        	else
        	{
        		fixed = false;
        	}
        }

        //Make start a class variable so you dont need to keep track
        void createAppendFile(std::vector<double>& data,const std::string datasetName,size_t start=0) {
		    H5::H5File file;
		    H5::DataSet dataset;
		    H5::DataSpace dataspace;

		    if(std::filesystem::exists(filename)) {
		    	if (datasetExists(filename,datasetName))
		    	{
		    		appendDataSet(data,datasetName,start);
		    	}
		    	else
		    	{
		    		// std::cerr<<"HERE"<<std::endl;
		    		createDataSet(data,datasetName,1);
		    		// std::cerr<<"HERE1"<<std::endl;
		    	}   
		    } else {
		        createDataSet(data,datasetName);
		    }
		}



        std::vector<double> readFile(const std::string datasetName) {
            std::vector<double> data;
            if(std::filesystem::exists(filename)) {
                H5::H5File file(filename, H5F_ACC_RDONLY);
                H5::DataSet dataset = file.openDataSet(datasetName);
                H5::DataSpace dataspace = dataset.getSpace();
                
                hsize_t dims_out[1];
                dataspace.getSimpleExtentDims(dims_out, NULL);
                data.resize(dims_out[0]);
                
                dataset.read(&data[0], H5::PredType::NATIVE_DOUBLE);
                file.close();
            }else{
            	std::cerr<<"File '"<<filename<<"' does not exist."<<std::endl;
            }
            return data;
        }

        void attachMetadataToDataset(const std::string& metadata, const std::string& datasetName) 
        {
		    // Initialize the HDF5 library
        	if(std::filesystem::exists(filename)) {
			    H5::H5File file(filename, H5F_ACC_RDWR);
	    		H5::DataSet dataset;
			    // Open the specified dataset
			    if (datasetExists(filename,datasetName)){
			    	dataset = file.openDataSet(datasetName);
			    }else{
			    	std::vector<double> dummy;
			    	dataset = createDataSet(dummy,datasetName);
			    }

			    // Create a string data type for the attribute
			    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);

			    // Check if the attribute's dataspace exists
			    if (attributeExists(dataset, "metadata")) {
			        std::cerr << "Attribute 'metadata' already exists for the dataset." << std::endl;
			        dataset.close();
			        file.close();
			        return;
			    }

			    // Create a dataspace for the attribute
			    H5::DataSpace attrSpace = H5::DataSpace(H5S_SCALAR);

			    // Create the attribute and write the metadata
			    H5::Attribute metadataAttr = dataset.createAttribute("metadata", strType, attrSpace);
			    metadataAttr.write(strType, metadata);

			    // Close resources
			    metadataAttr.close();
			    attrSpace.close();
			    dataset.close();
			    file.close();
		    }else{
            	std::cerr<<"File '"<<filename<<"' does not exist."<<std::endl;
            }
		}

		std::string readMetadataFromDataset(const std::string& datasetName) 
		{
		    std::string metadata;

		    // Initialize the HDF5 library
		    if(std::filesystem::exists(filename)) {
		    	H5::DataSet dataset;
                H5::H5File file(filename, H5F_ACC_RDONLY);
			    // Open the specified dataset
			    if (datasetExists(filename,datasetName)){
			    	dataset = file.openDataSet(datasetName);
			    }else{
			        std::cerr << "dataset '"<<datasetName<<"' does not exist for file '"<<filename<<"' ." << std::endl;
			    	return "ERROR RETRIEVING METADATA";
			    }

			    // Check if the attribute's dataspace exists
			    if (!attributeExists(dataset, "metadata")) {
			        std::cerr << "Attribute 'metadata' does not exist for dataset '"<<datasetName<<"' in file '"<<filename<<"' ." << std::endl;
			        dataset.close();
			        file.close();
			    	return "ERROR RETRIEVING METADATA";
			    }

			    // Open the metadata attribute
			    H5::Attribute metadataAttr = dataset.openAttribute("metadata");

			    // Read the metadata attribute
			    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
			    metadataAttr.read(strType, metadata);
			    
		    	// Close resources
			    metadataAttr.close();
			    dataset.close();
			    file.close();
            }else{
            	std::cerr<<"File '"<<filename<<"' does not exist."<<std::endl;
            }



		    return metadata;
		}


    private:
        std::string filename;
        size_t max_size;
        bool fixed;

        bool attributeExists(const H5::H5Object& object, const std::string& attributeName) 
        {
    		return object.attrExists(attributeName);
    	}


        bool datasetExists(const std::string& filename, const std::string& datasetName)
		{
		    // Open the HDF5 file
		    H5::H5File file(filename, H5F_ACC_RDONLY);

		    // Check if the dataset exists
		    return H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0;
		}

		H5::DataSet createDataSet(std::vector<double>& data,const std::string datasetName,int test=0)
		{
			H5::H5File file;
		    H5::DataSet dataset;
		    H5::DataSpace dataspace;
		    if (test ==1)
		    {
				file = H5::H5File(filename, H5F_ACC_RDWR);
		    }
		    else
		    {
				file = H5::H5File(filename, H5F_ACC_TRUNC);
		    }
	        hsize_t maxdims[1];
	        hsize_t dims[1] = {data.size()};
	        if (fixed)
	        {
	        	maxdims[0] = max_size; // Set maximum dimensions to max_size
	        	dataspace = H5::DataSpace(1, maxdims);
		        dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE,dataspace);
	        	dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);
	        }
	        else
	        {
	        	maxdims[0] = H5S_UNLIMITED; // Set maximum dimensions to unlimited
	        	dataspace = H5::DataSpace(1, dims, maxdims);
		        H5::DSetCreatPropList plist;
		        hsize_t chunk_dims[1] = {std::min((hsize_t)1000, (hsize_t)data.size())}; // Adjust chunk size as needed
		        plist.setChunk(1, chunk_dims);
		        dataset = file.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE, dataspace, plist);
	        	dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE);
	        }

	        file.close();
	        dataspace.close();
	        return dataset;
		}

		void appendDataSet(std::vector<double>& data,const std::string datasetName,size_t start=0)
		{
			H5::H5File file;
		    H5::DataSet dataset;
		    H5::DataSpace dataspace;
			file = H5::H5File(filename, H5F_ACC_RDWR);

	        dataset = file.openDataSet(datasetName);
	        dataspace = dataset.getSpace();

	        // Get current size of the dataset
	        hsize_t dims_out[1];
	        if (fixed)
	        {
	        	dims_out[0] = max_size;
	        }
	        else
	        {
	        	dataspace.getSimpleExtentDims(dims_out, NULL);
	        }

	        // Extend the dataset to hold the new data
	        hsize_t size[1] = {dims_out[0] + data.size()};
	        if (!fixed)
	        {
	        	dataset.extend(size);
	        }

	        // Select the extended part of the dataset
	        dataspace = dataset.getSpace();
	        hsize_t offset[1];
	        if (fixed)
	        {

	        	offset[0] = {start};
	        }
	        else
	        {

	        	offset[0] = {dims_out[0]};
	        }
	        hsize_t dimsextend[1] = {data.size()};
	        dataspace.selectHyperslab(H5S_SELECT_SET, dimsextend, offset);

	        // Write the data to the extended part of the dataset
	        H5::DataSpace memspace(1, dimsextend);
	        dataset.write(&data[0], H5::PredType::NATIVE_DOUBLE, memspace, dataspace);
	        file.close();
		}
};