/*
The MIT License

Copyright © 2024 Alexander P. Antonov, Sören Schweers, Artem Ryabov, and Philipp Maass

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the “Software”), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <curand_kernel.h>

std::string outputPrefix = "";                      /*prefix of output (folder name)*/
std::string cfgFile = "basep.cfg";                  /*default file name of configuration parameters*/

curandState *generatorStates;                       /*states of each CUDA threads RNG*/

//parallelization parameters
int ThreadsMAX = 1024;                              /*maximum number of threads per block*/
int cores = 1792;                                   /*number of cores of the GPU (hardware specific)*/
int numberOfGeneralizedIds;
int numberOfThreads;
int numberOfBlocks;
int *blockSize;
int *blockStart;
int *crossBorderCollisions;
bool *collisionHappened;
bool *anyCollisionHappened;
bool *anyCollisionHappenedHOST;

//simulation parameters
int N = 50;
int L = 100;
double sigma = 0.5;
double dt = 0.001;
double simulationDuration = 3000;
double equilibrationTime = 1000;
double order = 3;
double epsilon = 0.05;
double Gamma = 0;
double f = 1;
double U0 = 6;
double D = 1;

//state of simulation
double *x;
double *v;
int *identification;
bool *collision;

//measurements
double delta = 0.001;                               /*spatial resolition for measuring denity*/
int deltaSteps;

//results
double *densityProfile;
double *current;


__global__ void initializeRNG(curandState *generatorStates, int numberOfGeneralizedIds, int *seeds){
    //initialize rngs
    for(int i = 0; i<numberOfGeneralizedIds; i++){
        curand_init(seeds[i], i, 0, &generatorStates[i]);
    }
}

__global__ void initializeGPU(int *blockSize, int *blockStart, int numberOfGeneralizedIds, int *crossBorderCollisions, bool *collisionHappened, int N, int *identification, bool *collision, double *x, double *v, bool *anyCollisionHappened, double *current, double *densityProfile, int deltaSteps, int L){
    int remaining = N;

    for(int i = 0; i<numberOfGeneralizedIds; i++){
        blockSize[i] = __double2int_rn(remaining/(numberOfGeneralizedIds-i));
        remaining -= blockSize[i];
        crossBorderCollisions[i] = -1;
        collisionHappened[i] = false;
    }
    for(int i = numberOfGeneralizedIds; i<2*numberOfGeneralizedIds; i++){
        crossBorderCollisions[i] = -1;
    }
    if(remaining > 0){
        //std::cout<<remaining<<" remaining and added to last block (this should not be happening)"<<std::endl;
        blockSize[numberOfGeneralizedIds-1] += remaining;
    }

    blockStart[0] = 0;
    for(int i = 1; i<numberOfGeneralizedIds; i++){
        blockStart[i] = blockStart[i-1]+blockSize[i-1];
    }

    for(int i = 0; i < numberOfGeneralizedIds; i++){
        current[i] = 0;
    }

    for(int i = 0; i < N; i++){
        identification[i] = 1;
        collision[i] = false;
        x[i] = double(i*L)/double(N);
        v[i] = 0;
    }
    anyCollisionHappened[0] = false;

    for(int i = 0; i < deltaSteps * numberOfGeneralizedIds; i++){
        densityProfile[i] = 0;
    }
}

/**
 * @brief initialize blocks for simulation
 * 
 */
void establishBlocks(){
    cudaMalloc(&generatorStates, numberOfGeneralizedIds * sizeof(curandState));
    cudaMalloc(&blockSize, numberOfGeneralizedIds * sizeof(int));
    cudaMalloc(&blockStart, numberOfGeneralizedIds * sizeof(int));
    cudaMalloc(&crossBorderCollisions, 2 * numberOfGeneralizedIds * sizeof(int));
    cudaMalloc(&collisionHappened, numberOfGeneralizedIds * sizeof(bool));
    

    //initialize rngs
    srand (time(NULL));
    int *seeds;
    cudaMallocManaged(&seeds, numberOfGeneralizedIds * sizeof(int));
    for(int i = 0; i < numberOfGeneralizedIds; i++){
        seeds[i] = rand();
    }
    cudaDeviceSynchronize();
    initializeRNG<<<1,1>>>(generatorStates, numberOfGeneralizedIds, seeds);
    cudaDeviceSynchronize();
    cudaFree(seeds);
    initializeGPU<<<1,1>>>(blockSize, blockStart, numberOfGeneralizedIds, crossBorderCollisions, collisionHappened, N, identification, collision, x, v, anyCollisionHappened, current, densityProfile, deltaSteps, L);
}

/**
 * @brief update which clusters will definitely collide
 * 
 * @return __global__ 
 */
__global__ void updateCollisions(int *blockSize, int *blockStart, int *identification, double *v, double *x, bool *collision, int N, double dt, double sigma, int L) {
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    double dx;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        if(identification[pos]!=0 && identification[(pos+1)%N]!=0){
            dx = x[(pos+1)%N]-x[pos]-sigma;
            if(dx<0){
                dx += L;
            }
            if(v[pos] > v[(pos+1)%N] && (dx)/(v[pos]-v[(pos+1)%N]) < dt){
                collision[pos] = true;
            }else{
                collision[pos] = false;
            }
        }
    }
}

/**
 * @brief Get the minimum image distance
 * 
 * @param xi position of first particle
 * @param xj position of second particle
 * @return double 
 */
__device__ double getMinimumImageDistance(double xi, double xj, int L){
        double dx = xi - xj;
        if (dx < 0)
            dx = -dx;
        if (dx < 0.5 * L)
            return dx;
        else
            return L - dx;
}

/**
 * @brief Draw new particle velocities (random + deterministic forces)
 * 
 * @return arma::vec 
 */
__global__ void particleVelocities(double order, double epsilon, double Gamma, double sigma, int *blockSize, int *blockStart, double *v, double *x, int N, int L, double f, double U0, curandState *generatorStates, double dt, double D){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    double interaction = 0;
    double distance;

    int pos;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        interaction = 0;
        //get interaction forces, if particles are within interaction range epsilon
        
        if(Gamma > 0){
            distance = getMinimumImageDistance(x[pos], x[(pos-1+N)%N], L);
            if(distance < sigma + epsilon){
                interaction += -double(order)*Gamma*((pow(epsilon+sigma-distance, order-1))/((pow(epsilon, order+1)/(order+1))+Gamma*pow(epsilon+sigma-distance, order)));
            }
            distance = getMinimumImageDistance(x[(pos+1)%N], x[pos], L);
            if(distance < sigma + epsilon){
                interaction -= -double(order)*Gamma*((pow(epsilon+sigma-distance, order-1))/((pow(epsilon, order+1)/(order+1))+Gamma*pow(epsilon+sigma-distance, order)));
            }
        }
        
        //calculate particle velocity
        v[pos] = M_PI * U0 * sin(2 * M_PI * x[pos]) + f + interaction;
        v[pos] += sqrt(2*D/dt)*curand_normal_double(&generatorStates[generalizedId]);
    }
}

/**
 * @brief Calculate cluster velocities from culster identification and forces exterted on each particle
 * 
 * @return arma::vec 
 */
__global__ void clusterVelocities(int *blockSize, int *blockStart, double *v, int N, int *identification){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        if(identification[pos]>0){
            for(int j = 1; j < identification[pos]; j++){
                v[pos] += v[(pos+j)%N];
            }
            v[pos] /= identification[pos];

            for(int j = 1; j < identification[pos]; j++){
                v[(pos+j)%N] = v[pos];
            }
        }
    }
}

/**
 * @brief calculate center of mass (CM)
 * 
 * @param id 
 * @return double 
 */
__device__ double GPUgetClusterCM(int id, int *identification, double sigma, int N, double *x){
    id += 2*N;
    id = id%N;
    return x[id] * identification[id] + sigma * ((identification[id]-1) * identification[id]) / 2;
}

/**
 * @brief calculate center of mass (CM)
 * 
 * @param id 
 * @return double 
 */
double getClusterCM(int id, int *identification, double sigma, int N, double *x){
    id += 2*N;
    id = id%N;
    return x[id] * identification[id] + sigma * ((identification[id]-1) * identification[id]) / 2;
}

/**
 * @brief Get the Cluster C M object
 * 
 * @param id 
 * @return double 
 */
__device__ double getClusterV(int id, int *identification, int N, double *v){
    id += 2*N;
    id = id%N;
    return v[id] * identification[id];
}

/**
 * @brief Get the Cluster C M object
 * 
 * @param id 
 * @return double 
 */
__device__ double getClusterSize(int id, int *identification, int N){
    id += 2*N;
    id = id%N;
    return identification[id];
}

__device__ int getLeftId(int id, int *identification, int N){
    if(identification[(id-1+N)%N]<0){
        return identification[(id-1+N)%N] + id;
    }else{
        return id - 1;
    }
}

/**
 * @brief merge clusters within block boundary
 * 
 * @return __global__ 
 */
__global__ void merge(int *blockSize, int *blockStart, int *identification, double *v, double *x, bool *collision, int N, double dt, double sigma, int L, bool *collisionHappened, int *crossBorderCollisions, int numberOfGeneralizedIds) { 
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    int first;
    int last;
    int size;
    double speed;
    double cm;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        if(collision[pos]){
            collision[pos] = false;
            last = identification[(pos+1)%N]+pos;
            if(identification[pos]<0){
                first = pos+identification[pos]+1;
            }else{
                first = pos;
            }

            if(last < blockStart[generalizedId]+blockSize[generalizedId] && first >= blockStart[generalizedId] ){
                cm = GPUgetClusterCM(first, identification, sigma, N, x);
                cm += GPUgetClusterCM((pos+1)%N, identification, sigma, N, x);
                if(x[(pos+1)%N]<x[(first+N)%N]){
                    cm += L*identification[(pos+1)%N];
                }
                size = identification[(first+N)%N]+identification[(pos+1)%N];
                speed = identification[(first+N)%N]*v[(first+N)%N] + identification[(pos+1)%N]*v[(pos+1)%N];

                speed /= size;
                cm /= size;
                for(int j = first; j <= last; j++){
                    v[(j+N)%N] = speed;
                    x[(j+N)%N] = cm-sigma*(double(size-1)/2.0-double(j-first));
                    identification[(j+N)%N] = 0;
                }
                identification[(first+N)%N] = size;
                identification[last%N] = -size;
                collisionHappened[generalizedId] = true;
            }else if (last >= blockStart[generalizedId]+blockSize[generalizedId]){
                crossBorderCollisions[generalizedId] = pos;
            }else{
                crossBorderCollisions[generalizedId+numberOfGeneralizedIds] = pos;
            } 
        }
    }
}

/**
 * @brief merge clusters across block boundary
 * 
 * @return true 
 * @return false 
 */
__global__ void mergeCrossBlockBorders(bool *anyCollisionHappened, int *crossBorderCollisions, int *identification, int L, int N, double *x, double *v, double sigma, bool *collisionHappened, int numberOfGeneralizedIds){
    int pos;
    int first;
    int last;
    int size;
    double speed;
    double cm;
    anyCollisionHappened[0] = false;
    for(int i = 0; i < 2*numberOfGeneralizedIds; i++){
        if(crossBorderCollisions[i] > -1){
            pos = crossBorderCollisions[i];
            anyCollisionHappened[0] = true;
            crossBorderCollisions[i] = -1;

            last = identification[(pos+1)%N]+pos;
            if(identification[pos]<0){
                first = pos+identification[pos]+1;
            }else{
                first = pos;
            }

            cm = GPUgetClusterCM((first+N)%N, identification, sigma, N, x);
            cm += GPUgetClusterCM((pos+1)%N, identification, sigma, N, x);
            if(x[(pos+1)%N]<x[(first+N)%N]){
                cm += L*identification[(pos+1)%N];
            }
    
            size = identification[(first+N)%N]+identification[(pos+1)%N];
            speed = identification[(first+N)%N]*v[(first+N)%N] + identification[(pos+1)%N]*v[(pos+1)%N];
            
            speed /= size;
            cm /= size;
            for(int j = first; j <= last; j++){
                v[(j+N)%N] = speed;
                x[(j+N)%N] = cm-sigma*(double(size-1)/2.0-double(j-first));
                identification[(j+N)%N] = 0;
            }
            
            identification[(first+N)%N] = size;
            identification[last%N] = -size;
        }
        if(collisionHappened[i]){
            collisionHappened[i] = false;
            anyCollisionHappened[0] = true;
        }
    }
}

/**
 * @brief Maps the particle position back into the central image box according to the boundary conditions.
 * 
 */
__global__ void applyPeriodicBoundaryConditions(int *blockSize, int *blockStart, int L, double *x){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        // Apply periodic boundary conditions
        if(x[pos] >= L){
            x[pos] -= L;
        }else if(x[pos] < 0.0){
            x[pos] += L;
        }
    }
}

/**
 * @brief Maps the particle position back into the central image box according to the boundary conditions.
 * 
 */
__global__ void advanceTime(int *blockSize, int *blockStart, double *x, double *v, double *current, double dt, int L){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        x[pos] += v[pos]*dt; 
        current[generalizedId] += v[pos]*dt/L;
    }
}

/**
 * @brief Splitting clusters at the correct positions
 * 
 */
__global__ void splitClusters(int *blockSize, int *blockStart, double *v, double *x, int *identification, int N){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    double forceLeft;                                   /*total force on the left subcluster*/
    double forceRight;                                  /*total force on the left subcluster*/

    int relativePosOfBiggestDifference;                 /*position of potential split*/
    double lowestInteractionForce;                      /*lowest difference of the interaction forces of the two subclusters found so far*/

    int pos;

    //iterate over all particles in the block
    for(int i = 0; i < blockSize[generalizedId]; i++){
        //id of the first particle of the next cluster
        pos = i + blockStart[generalizedId];

        //if the particle is the first one of a cluster, we check if the cluster should be split
        if(identification[pos]>1){
            forceLeft = 0;
            forceRight = 0;

            relativePosOfBiggestDifference = -1;
            lowestInteractionForce = 0;

            //calculate total force exerted on the cluster
            for(int j = 0; j < identification[pos]; j++){
                forceRight += v[(pos+j)%N];
            }

            //calculate forces on two subclusters for all possible splitting positions
            for(int j = 0; j < identification[pos]-1; j++){
                forceRight -= v[(pos+j)%N];
                forceLeft += v[(pos+j)%N];

                //find the pair of subclusters for which the velocity difference is the largest
                if(forceLeft/(j+1) - forceRight/(identification[pos]-j-1) < lowestInteractionForce){
                    lowestInteractionForce = forceLeft/(j+1) - forceRight/(identification[pos]-j-1);
                    relativePosOfBiggestDifference = j;
                }      
            }

            //if the difference is positive, we split the cluster by updateing the configuration
            if(relativePosOfBiggestDifference > -1){
                identification[(pos+relativePosOfBiggestDifference+1)%N] = identification[pos] - relativePosOfBiggestDifference - 1;
                identification[pos] = relativePosOfBiggestDifference + 1;  

                //check the new cluster again
                i--;

                if(identification[pos] > 1){
                    identification[(pos+identification[pos]-1)%N] = - identification[pos];
                }

                if(identification[(pos+relativePosOfBiggestDifference+1)%N] > 1){
                    identification[(pos+relativePosOfBiggestDifference+1+identification[(pos+relativePosOfBiggestDifference+1)%N]-1)%N] = - identification[(pos+relativePosOfBiggestDifference+1)%N];
                }
            } 
        }
    }
}

/**
 * @brief Initialize simulation
 * 
 */
void initialize(){
    int device;                                     /*device id of the device in use*/
    cudaDeviceProp prop;                            /*properties of the device in use*/
    cudaGetDevice(&device);                         /*get device id*/
    cudaGetDeviceProperties(&prop, device);         /*get device properties*/
    cores = prop.multiProcessorCount*prop.warpSize; /*calculate number of cores*/
    ThreadsMAX = prop.maxThreadsPerBlock;           /*get maximum number of threads per block*/
    
    //choose number of cores and threads per block
    numberOfGeneralizedIds = int(N/5);
    if(numberOfGeneralizedIds > cores){
        numberOfGeneralizedIds = cores;
    }
    numberOfBlocks = int(std::ceil(double(numberOfGeneralizedIds)/double(ThreadsMAX)));
    numberOfThreads = numberOfGeneralizedIds/numberOfBlocks;
    numberOfGeneralizedIds = numberOfBlocks * numberOfThreads;

    //print device properties
    std::cout<<""<<std::endl;
    std::cout<<"Device Name: "<<prop.name<< std::endl;
    std::cout<<"Device Version: "<<prop.major<<"."<<prop.minor<< std::endl;
    std::cout<<"Number SM = "<<prop.multiProcessorCount<<std::endl;
    std::cout<<"Warp Size: "<<prop.warpSize<<std::endl;
    std::cout<<"Number of cores: "<<cores<<std::endl;
    std::cout<<"Max Threads Per Block: "<<prop.maxThreadsPerBlock<<std::endl;
    std::cout<<"Total Global Memory: "<< prop.totalGlobalMem<<std::endl;
    std::cout<<""<<std::endl;
    std::cout<<"Number of Blocks in use: "<<numberOfBlocks<<std::endl;
    std::cout<<"Threads per Block in use: "<<numberOfThreads<<std::endl;
    std::cout<<"Number of Cores in use: "<<numberOfGeneralizedIds<<std::endl;
    std::cout<<""<<std::endl;

    //calculate resolution of density profile
    deltaSteps = round(1/delta);

    //reserve memory for simulation
    cudaMallocManaged(&densityProfile, numberOfGeneralizedIds * deltaSteps * sizeof(double));
    cudaMallocManaged(&current, numberOfGeneralizedIds * sizeof(double));
    cudaMalloc(&anyCollisionHappened, sizeof(bool));
    anyCollisionHappenedHOST = (bool*)malloc(1);
    anyCollisionHappenedHOST[0] = false;
    cudaMalloc(&x, N * sizeof(double));
    cudaMalloc(&v, N * sizeof(double));
    cudaMalloc(&identification, N * sizeof(int));
    cudaMalloc(&collision, N * sizeof(bool));
    cudaDeviceSynchronize();
    establishBlocks();
    cudaDeviceSynchronize();
}

/**
 * @brief Clean up memory after execution of the simulation
 * 
 */
void cleanUp(){
    cudaFree(x);
    cudaFree(v);
    cudaFree(identification);
    cudaFree(collision);
    cudaFree(blockSize);
    cudaFree(blockStart);
    cudaFree(collisionHappened);
    cudaFree(crossBorderCollisions);
    cudaFree(generatorStates);
    cudaFree(anyCollisionHappened);
    cudaFree(current);
    cudaFree(densityProfile);
}

/**
 * @brief Print the current state of the simulation
 * 
 */
void printState(){
    for(int i = 0; i < N; i++){
        std::cout<<"x["<<i<<"]="<<x[i]<<std::endl;
        std::cout<<"v["<<i<<"]="<<v[i]<<std::endl;
        std::cout<<"collision["<<i<<"]="<<collision[i]<<std::endl;
        std::cout<<"identification["<<i<<"]="<<identification[i]<<std::endl;
        std::cout<<" "<<std::endl;
    }
}

/**
 * @brief Perform measurement of the density profile
 * 
 */
__global__ void measureDensity(int *blockStart, int *blockSize, double *densityProfile, double *x, int deltaSteps){
    int generalizedId = blockIdx.x*blockDim.x+threadIdx.x;
    int pos;
    for(int i = 0; i < blockSize[generalizedId]; i++){
        pos = i + blockStart[generalizedId];
        densityProfile[int(round((x[pos] - floor(x[pos]) + generalizedId) * deltaSteps))] += 1;
    }
}

/**
 * @brief Perform one step of the simulation
 * 
 */
void doStep(){
    //calculate new particle velocities
    particleVelocities<<<numberOfBlocks,numberOfThreads>>>(order, epsilon, Gamma, sigma, blockSize, blockStart, v, x, N, L, f, U0, generatorStates, dt, D);
    cudaDeviceSynchronize();

    //split clusters according to the forces exerted on the particles
    splitClusters<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, v, x, identification, N);

    //update cluster velocities
    clusterVelocities<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, v, N, identification);
    do{
        cudaDeviceSynchronize();

        //update which clusters will definitely collide
        updateCollisions<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, identification, v, x, collision, N, dt, sigma, L);
        cudaDeviceSynchronize();

        //merge colliding clusters within block boundary
        merge<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, identification, v, x, collision, N, dt, sigma, L, collisionHappened, crossBorderCollisions, numberOfGeneralizedIds);
        cudaDeviceSynchronize();

        //merge colliding clusters across block boundary
        mergeCrossBlockBorders<<<1,1>>>(anyCollisionHappened, crossBorderCollisions, identification, L, N, x, v, sigma, collisionHappened, numberOfGeneralizedIds);
        cudaDeviceSynchronize();

        //check if any collision happened
        cudaMemcpy(anyCollisionHappenedHOST, anyCollisionHappened, 1, cudaMemcpyDeviceToHost);
    }while(anyCollisionHappenedHOST[0]);
    cudaDeviceSynchronize();

    //advance time by dt, propagate particles
    advanceTime<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, x, v, current, dt, L);
    cudaDeviceSynchronize();

    //apply periodic boundary conditions
    applyPeriodicBoundaryConditions<<<numberOfBlocks,numberOfThreads>>>(blockSize, blockStart, L, x);
    cudaDeviceSynchronize();

    //measure density profile
    measureDensity<<<numberOfBlocks,numberOfThreads>>>(blockStart, blockSize, densityProfile, x, deltaSteps);
    cudaDeviceSynchronize();
}

/**
 * @brief Simulate the system for the given duration while performing measurements
 * 
 */
void simulateSystem(){
    cudaDeviceSynchronize();
    double time = 0.0;
    while(time < simulationDuration){
        doStep();
        time += dt;
    }
    cudaDeviceSynchronize();
}

/**
 * @brief Simulate the system for the given duration without performing measurements
 * 
 */
void getEquilibriumState(){
    cudaDeviceSynchronize();
    double time = 0.0;
    while(time < equilibrationTime){
        doStep();
        time += dt;
    }
    cudaDeviceSynchronize();

    //reset measurements
    for(int i = 0; i < numberOfGeneralizedIds; i++){
        current[i] = 0;
    }
    for(int i = 0; i < deltaSteps * numberOfGeneralizedIds; i++){
        densityProfile[i] = 0;
    }
    cudaDeviceSynchronize();
}

/**
 * @brief Save the results of the simulation to output files
 * 
 */
void saveResults(){
    //prepare output files
    boost::filesystem::create_directory("output/");
    boost::filesystem::create_directory(outputPrefix);
    FILE *currentFile;
    FILE *DensityFile;
    std::string filename = outputPrefix+"/current";
    currentFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/density";
    DensityFile = fopen(filename.c_str(), "w");

    //get density from all blocks
    for(int i = deltaSteps; i<deltaSteps*numberOfGeneralizedIds; i++){
        densityProfile[i%deltaSteps] += densityProfile[i];
    }

    //save particle density
    for(int i = 0; i < deltaSteps; ++i){
        fprintf(DensityFile, "%f, ", i * delta);
        fprintf(DensityFile, "%f\n", densityProfile[i]/(simulationDuration/dt*L*delta));
    }

    //get particle current from all blocks
    for(int i = 1; i < numberOfGeneralizedIds; i++){
        current[0] += current[i];
    }

    //save particle current
    fprintf(currentFile, "%f", current[0]/simulationDuration);

    //close output files
    fclose(currentFile);
    fclose(DensityFile);
}

/**
 * @brief Load configuration file
 * 
 */
void loadInput(){
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    
    std::ifstream file;
    file.open(cfgFile);

    desc.add_options()
    ("help", "Print help messages") 
    ("totaltime,t", po::value<double>(&simulationDuration), "Total time of simulation.")
    ("particles,N", po::value<int>(&N), "Number of particles.")
    ("length,L", po::value<int>(&L), "Length of the simulated system.")
    ("diameter,sigma", po::value<double>(&sigma), "Diameter of hard rods.")
    ("DiffusionConstant,D", po::value<double>(&D), "Diffusion constant.")
    ("timestep,dt", po::value<double>(&dt), "Timestep for Langevin-Euler-Scheme.")
    ("force,f", po::value<double>(&f), "Amplitude of constants drag force.")
    ("AmpU,U", po::value<double>(&U0), "Amplitude of external periodic potential.")
    ("gamma,stickyness", po::value<double>(&Gamma), "Adhesive interaction strength.")
    ("order", po::value<double>(&order), "Polynomial order of the representation of the delta function.")
    ("epsilon", po::value<double>(&epsilon), "Range of the representation of the delta function.")
    ("eqtime", po::value<double>(&equilibrationTime), "Equilibration time before sampling starts.")
    ("delta", po::value<double>(&delta), "Spatial resolution.")
    ("prefix,o", po::value<std::string>(&outputPrefix), "Path prefix for output files.")
    ;

    po::variables_map vm;
    po::store(boost::program_options::parse_config_file(file,desc),vm);
    po::notify(vm);

    std::cout<<"totaltime: "<<simulationDuration<<std::endl;
    std::cout<<"N = "<<N<<std::endl;
    std::cout<<"L = "<<L<<std::endl;
    std::cout<<"sigma = "<<sigma<<std::endl;
    std::cout<<"D = "<<D<<std::endl;
    std::cout<<"dt = "<<dt<<std::endl;
    std::cout<<"f = "<<f<<std::endl;
    std::cout<<"U0 = "<<U0<<std::endl;
    std::cout<<"gamma = "<<Gamma<<std::endl;
    std::cout<<"order = "<<order<<std::endl;
    std::cout<<"epsilon = "<<epsilon<<std::endl;
    std::cout<<"eqtime: "<<equilibrationTime<<std::endl;
    std::cout<<"delta = "<<delta<<std::endl;
    std::cout<<"prefix: "<<outputPrefix<<std::endl;

    outputPrefix = "output/" + outputPrefix;
}

/**
 * @brief Simulate Baxter's adhesive hard spheres and 
 * pure hard spheres (gamma = 0)
 * 
 * @param argc 
 * @param input path to configuration file
 * @return int 
 */
int main(int argc, char** input){

    //check if a configuration file is provided
    if(argc > 1){
        cfgFile = input[1];
    }

    //load configuration file
    loadInput();

    //initialize the simulation
    initialize();

    //run simulation until equilibrium is reached
    getEquilibriumState();

    //save start time
    auto start = std::chrono::system_clock::now();

    //run simulation and perform measurements
    simulateSystem();

    //save end time
    auto end = std::chrono::system_clock::now();

    //calculate the computation time
    std::chrono::duration<double> elapsed_seconds = end-start;

    //print computation time
    std::cout<<elapsed_seconds.count()<<" s"<<std::endl;

    //save results
    saveResults();

    //clean up memory
    cleanUp();
    return 0;
}

