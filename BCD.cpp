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
#include <iostream>
#include <cmath>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/filesystem.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <thread>
#include <vector>

std::string outputPrefix = "";                      /*prefix of output (folder name)*/
std::string cfgFile = "basep.cfg";                  /*default file name of configuration parameters*/

//generator for seeds for all threads
boost::random::mt19937 SeedGenerator((std::chrono::system_clock::now().time_since_epoch()).count());

//generator for random numbers (uniform distribution [0, 1])
boost::random::mt19937 Generator((std::chrono::system_clock::now().time_since_epoch()).count());
boost::variate_generator<boost::mt19937&, boost::uniform_01<> > randUniform(Generator, boost::uniform_01<>());

//parallelization parameters
int ThreadsMAX;                                     /*number of available threads*/      
int numberOfBlocks;                                 /*number of blocks used for parallelization*/
int *blockSize;                                     /*size of each block*/
int *blockStart;                                    /*start of each block*/
int *crossBorderCollisions;                         /*clusters that will collide across block borders*/
bool *collisionHappened;                            /*did a collision happened?*/
int *blockStartsForSplitting;                       /*start of each block for the fragmentation procedure*/

//simulation parameters
int N;                                              /*particles number*/
int L;                                              /*system length*/       
double sigma;                                       /*particle size*/
double dt;                                          /*time step*/
double simulationDuration;                          /*total simulation time*/
double equilibrationTime;                           /*time to run th esimulation before starting measurements*/
double order;                                       /*order of polynomial representation of delta function*/
double epsilon;                                     /*range of adhesive interaction*/
double Gamma;                                       /*amplitude of adhesive interaction*/
double f;                                           /*constant drag force*/
double U0;                                          /*amplitude of cosine potentail*/
double D;                                           /*diffusion coefficient*/

//state of simulation
double *x;                                          /*particle positions*/            
double *v;                                          /*particle velocities*/
int *identification;                                /*cluster identification:   1 = independent particle
                                                                                0 = particle is inside of a cluster
                                                                                2, 3, .., N = first particle of a cluster with this length
                                                                                -2, -3, .., -N = last particle of a cluster with this length*/
bool *collision;                                    /*collision between particles?*/

//measurements
double delta = 0.001;                               /*spatial resolition for measuring denity*/
double deltaPrime = delta;                          /*spatial resolition for measuring two-particle-density at contact*/
double tfTrajectory = 1000;                         /*time between saving the particle positions*/
double tfCurrent = 1000;                            /*time between saving the particle current*/
double tfMSD = 0.001;                               /*time between saving the mean square displacement*/
int spatialSteps;                                   /*spatial resolution when measuring the density*/
int deltaPrimeSteps;                                /*spatial resolution when measuring two-particle densities at contact*/

//results
double *density;                                    /*one-partcile density*/
double *density_II;                                 /*two-particle density at contact*/
double *current;                                    /*particle current*/
double *MSD;                                        /*mean square displacement*/
double *x_initial;                                  /*initial particle positions*/
double *x_noPB;                                     /*particle positions without periodic boundaries*/
double *x_old;                                      /*old particle positions*/ 


/**
 * @brief Initialize blocks and initial configuration of the system
 * 
 */
void establishBlocks(){
    blockSize = (int*)malloc(sizeof(int)*numberOfBlocks);
    blockStart = (int*)malloc(sizeof(int)*numberOfBlocks);
    crossBorderCollisions = (int*)malloc(sizeof(int)*numberOfBlocks*2);
    collisionHappened = (bool*)malloc(sizeof(bool)*numberOfBlocks);
    blockStartsForSplitting = (int*)malloc(sizeof(int)*numberOfBlocks);

    //assign particles to blocks
    int remaining = N;
    for(int i = 0; i<numberOfBlocks; i++){
        blockSize[i] = int(round(remaining/(numberOfBlocks-i)));
        remaining -= blockSize[i];
        crossBorderCollisions[i] = -1;
        collisionHappened[i] = false;
    }
    for(int i = numberOfBlocks; i<2*numberOfBlocks; i++){
        crossBorderCollisions[i] = -1;
    }
    //add remaining particles to last block
    if(remaining > 0){
        std::cout<<remaining<<" remaining and added to last block (this should not be happening)"<<std::endl;
        blockSize[numberOfBlocks-1] += remaining;
    }

    //calculate start of each block
    blockStart[0] = 0;
    for(int i = 1; i<numberOfBlocks; i++){
        blockStart[i] = blockStart[i-1]+blockSize[i-1];
    }

    //initialize particles
    for(int i = 0; i < N; i++){
        identification[i] = 1;
        collision[i] = false;
        v[i] = 0;
    }

    /*Windows for each particle position are calculated so that there are no conflicts.
    Particle positions are put in each window according to a uniform distribution*/
    for (int i = 0; i < N; ++i) {
        x[i] = double(L)/double(N) * i + sigma/2 + randUniform() * (double(L)/double(N)-sigma);
        x_old[i] = x[i];
        x_initial[i] = x[i];
        x_noPB[i] = x[i];
    }
}

/**
 * @brief Update which clusters will definitely collide
 * 
 */
void updateCollisions(int id) {
    int pos;
    double dx;
    //go through all particles in block and check if they will collide
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        if(identification[pos]!=0 && identification[(pos+1)%N]!=0){
            dx = x[(pos+1)%N]-x[pos];
            if(dx<0){
                dx += L;
            }
            if(v[pos] > v[(pos+1)%N]){
                if((dx-sigma)/(v[pos]-v[(pos+1)%N]) < dt){
                    collision[pos] = true;
                }else{
                    collision[pos] = false;
                }
            }
        }
    }
}

/**
 * @brief Get the minimum image distance between two particles
 * 
 * @param xi position of first particle
 * @param xj position of second particle
 * @return double 
 */
double getMinimumImageDistance(double xi, double xj){
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
 */
void particleVelocities(int id, int seed){
    //initialize local RNG of each block
    boost::random::mt19937 Generator(seed);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > randNormal(Generator, boost::normal_distribution<>());

    double interaction = 0;
    double distance;

    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        interaction = 0;
        //get interaction forces, if particles are within interaction range epsilon
        if(Gamma > 0){
            distance = getMinimumImageDistance(x[pos], x[(pos-1+N)%N]);
            if(distance < sigma + epsilon){
                interaction += -double(order)*Gamma*((pow(epsilon+sigma-distance, order-1))/((pow(epsilon, order+1)/(order+1))+Gamma*pow(epsilon+sigma-distance, order)));
            }
            distance = getMinimumImageDistance(x[(pos+1)%N], x[pos]);
            if(distance < sigma + epsilon){
                interaction -= -double(order)*Gamma*((pow(epsilon+sigma-distance, order-1))/((pow(epsilon, order+1)/(order+1))+Gamma*pow(epsilon+sigma-distance, order)));
            }
        }
        
        //calculate particle velocity
        v[pos] = boost::math::double_constants::pi * U0 * sin(2 * boost::math::double_constants::pi * x[pos]) + f + interaction;
        v[pos] += sqrt(2*D/dt)*randNormal();
    }
}

/**
 * @brief Calculate cluster velocities from culster identification and particle velocities
 * 
 * @param id of blocks
 * 
 */
void clusterVelocities(int id){
    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
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
 * @brief Total sum of all positions of the particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @param identification of clusters
 * @param sigma size of the particles
 * @param N number of particles
 * @param x particle positions
 * 
 * @return double 
 */
double getClusterCM(int id, int *identification, double sigma, int N, double *x){
    id += 2*N;
    id = id%N;
    return x[id] * identification[id] + sigma * ((identification[id]-1) * identification[id]) / 2;
}

/**
 * @brief Total sum of all velocities of the particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @param identification of clusters
 * @param N number of particles
 * @param v particle velocities
 * 
 * @return double 
 */
double getClusterV(int id, int *identification, int N, double *v){
    id += 2*N;
    id = id%N;
    return v[id] * identification[id];
}

/**
 * @brief Number of particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @param identification of clusters
 * @param N number of particles
 * 
 * @return double 
 */
double getClusterSize(int id, int *identification, int N){
    id += 2*N;
    id = id%N;
    return identification[id];
}

/**
 * @brief Get the id of the next cluster to the left
 * 
 * @param id of the first particle of the cluster
 * @param identification of clusters
 * @param N number of particles
 * 
 * @return int 
 */
int getLeftId(int id, int *identification, int N){
    if(identification[(id-1+N)%N]<0){
        return identification[(id-1+N)%N] + id;
    }else{
        return id - 1;
    }
}

void printState();

/**
 * @brief Check if the cluster identification is correct
 * 
 * @param where "where" in the code the check is performed
 */
void checkIdentification(std::string where){
    int lengthOfCluster = 0;
    int remaining = 0;
    int offset = 0;

    //search for the first particle of a cluster
    while(identification[offset]<1){
        offset++;
    }

    //check the cluster identification for all particles
    for(int i = 0; i < N; i++){
        if(remaining > 0){
            remaining--;
            if(remaining==0){
                if(identification[(offset+i)%N]!=-lengthOfCluster){
                    printState();
                    std::cout<<std::endl;
                    std::cout<<where<<std::endl;
                    std::cout<<(offset+i)%N<<"should be end of cluster"<<std::endl;
                    throw 1;
                }
            }else{
                if(identification[(offset+i)%N]!=0){
                    printState();
                    std::cout<<std::endl;
                    std::cout<<where<<std::endl;
                    std::cout<<(offset+i)%N<<"should be 0"<<std::endl;
                    throw 1;
                }
            }
        }else{
            if(identification[(offset+i)%N]>1){
                lengthOfCluster = identification[(offset+i)%N];
                remaining = identification[(offset+i)%N]-1;
            }
            if(identification[(offset+i)%N]<=0){
                printState();
                std::cout<<std::endl;
                std::cout<<where<<std::endl;
                std::cout<<(offset+i)%N<<"should be start of cluster"<<std::endl;
                throw 1;
            }
        }
    }
}

/**
 * @brief Merge clusters within block boundaries
 * 
 * @param id of the block
 * 
 */
void merge(int id) { 
    //save data of current cluster
    int pos;
    int first;
    int last;
    int size;
    double speed;
    double cm;

    //go through all particles in block and check if they will collide
    for(int i = 0; i < blockSize[id]-1; i++){
        //id of current cluster (first particle of the cluster)
        pos = i + blockStart[id];

        //check if the cluster will collide
        if(collision[pos]){
            collision[pos] = false;
            last = identification[(pos+1)%N]+pos;
            if(identification[pos]<0){
                first = pos+identification[pos]+1;
            }else{
                first = pos;
            }

            //if the cluster is not at the border of the block, perform the merge
            if(last < blockStart[id]+blockSize[id] && first >= blockStart[id]){
                cm = getClusterCM(first, identification, sigma, N, x);
                cm += getClusterCM((pos+1)%N, identification, sigma, N, x);
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
                collisionHappened[id] = true;
            
            //if the cluster is at the border of the block, save the collision for later
            }else if (last >= blockStart[id]+blockSize[id]){
                crossBorderCollisions[id] = pos;
            }else{
                crossBorderCollisions[id+numberOfBlocks] = pos;
            } 
        }
    }

    //if the cluster is at the border of the block, save the collision for later
    if(collision[blockStart[id]+blockSize[id]-1]){
        collision[blockStart[id]+blockSize[id]-1] = false;
        crossBorderCollisions[id] = blockStart[id]+blockSize[id]-1;
    }
}

/**
 * @brief Merge clusters across block boundary
 * 
 * @return true a collision happened
 * @return false no collision happened
 * 
 */
bool mergeCrossBlockBorders(){
    //save data of current cluster
    int pos;
    int first;
    int last;
    int size;
    double speed;
    double cm;
    bool anyCollisionHappened = false;

    //go through all collisions that happened across block borders
    for(int i = 0; i < 2*numberOfBlocks; i++){

        //if a collision is detected, merge the clusters
        if(crossBorderCollisions[i] > -1){
            pos = crossBorderCollisions[i];

            //remember that a collision happened
            anyCollisionHappened = true;
            crossBorderCollisions[i] = -1;

            last = identification[(pos+1)%N]+pos;
            if(identification[pos]<0){
                first = pos+identification[pos]+1;
            }else{
                first = pos;
            }

            cm = getClusterCM((first+N)%N, identification, sigma, N, x);
            cm += getClusterCM((pos+1)%N, identification, sigma, N, x);
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
    }

    //remember if a collision happened
    for(int i = 0; i < numberOfBlocks; i++){
        if(collisionHappened[i]){
            collisionHappened[i] = false;
            anyCollisionHappened = true;
        }
    }
    return anyCollisionHappened;
}

/**
 * @brief Maps the particle position back into the central image box according to the boundary conditions.
 * 
 * @param id of the block
 * 
 */
void applyPeriodicBoundaryConditions(int id){
    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        // Apply periodic boundary conditions
        if(x[pos] >= L){
            x[pos] -= L;
        }else if(x[pos] < 0.0){
            x[pos] += L;
        }
    }
}

/**
 * @brief Advance time by one time step
 * 
 * @param id of the block
 * 
 */
void advanceTime(int id){
    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];

        //particle propagation
        x[pos] += v[pos]*dt;

        if(x[pos]-x_old[pos]>L/2){
            x_old[pos]+=L;
        }
        if(x[pos]-x_old[pos]<-L/2){
            x_old[pos]-=L;
        }
        x_noPB[pos] += x[pos]-x_old[pos]; 
        x_old[pos] = x[pos];
    }
}

/**
 * @brief Find a good starting position for the fragmentation procedure for each block (i.e. beginning of a cluster)
 * 
 * @param id of the block
 * 
 */
void startOfSplits(int id){
    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        if(identification[pos] > 0){
            blockStartsForSplitting[id] = pos;
            return;
        }
    }
    blockStartsForSplitting[id] = -1;
}

/**
 * @brief Splitting clusters at the correct positions
 * 
 * @param id of the block
 * 
 */
void splitClusters(int id){
    double forceLeft;                           /*total force on the left subcluster*/
    double forceRight;                          /*total force on the right subcluster*/

    int relativePosOfBiggestDifference;         /*position of potential split*/
    double lowestInteractionForce;              /*lowest difference of the interaction forces of the two subclusters found so far*/

    int pos;

    //exit if no cluster were assigned to the block
    int blockSize;
    if(blockStartsForSplitting[id]<0){
        return;
    }

    //find the next block to which clusters were assigned
    int k = (id + 1)%numberOfBlocks;
    while(blockStartsForSplitting[k]<0){
        k = (k+1)%numberOfBlocks;
    }

    //calculate the size of the block
    blockSize = blockStartsForSplitting[k]-blockStartsForSplitting[id];
    if(blockSize<=0){
        blockSize += N;
    }
    
    //iterate over all particles in the block
    for(int i = 0; i < blockSize; i++){
        //id of the first particle of the next cluster
        pos = (i + blockStartsForSplitting[id])%N;

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
    //get number of available threads on the machine
    ThreadsMAX = std::thread::hardware_concurrency();

    //redduce the number of threads if the number of particles is small
    numberOfBlocks = int(N/1000);
    if(numberOfBlocks > ThreadsMAX){
        numberOfBlocks = ThreadsMAX;
    }

    //set a maximum number of threads
    //if(numberOfBlocks > 4){
    //    numberOfBlocks = 4;
    //}

    //use at least one thread
    if(numberOfBlocks < 1){
        numberOfBlocks = 1;
    }

    //print information about the number of threads available and used
    std::cout<<""<<std::endl;
    std::cout<<"Number of available Threads: "<<ThreadsMAX<<std::endl;
    std::cout<<"Number of used Threads: "<<numberOfBlocks<<std::endl;
    std::cout<<""<<std::endl;


    //update resolition for density measurements
    spatialSteps = round(1/delta);
    deltaPrimeSteps = round(1/deltaPrime);

    x = (double*)malloc(sizeof(double)*N);
    x_initial = (double*)malloc(sizeof(double)*N);
    x_noPB = (double*)malloc(sizeof(double)*N);
    x_old = (double*)malloc(sizeof(double)*N);
    v = (double*)malloc(sizeof(double)*N);
    identification = (int*)malloc(sizeof(int)*N);
    collision = (bool*)malloc(sizeof(bool)*N);
    
    //initialize all blocks
    establishBlocks();

    //prepare measurements
    density = (double*)malloc(sizeof(double)*spatialSteps*numberOfBlocks);
    density_II = (double*)malloc(sizeof(double)*spatialSteps*numberOfBlocks);
    for(int i = 0; i<spatialSteps*numberOfBlocks; i++){
        density[i] = 0;
        density_II[i] = 0;
    }
    current = (double*)malloc(sizeof(double)*numberOfBlocks);
    MSD = (double*)malloc(sizeof(double)*numberOfBlocks);
    for(int i = 0; i < numberOfBlocks; i++){
        current[i] = 0;
        MSD[i] = 0;
    }
}

/**
 * @brief Clean up memory after execution of the simulation
 * 
 */
void cleanUp(){
    free(x);
    free(x_initial);
    free(x_noPB);
    free(x_old);
    free(v);
    free(identification);
    free(collision);
    free(collisionHappened);
    free(crossBorderCollisions);
    free(blockSize);
    free(blockStart);
    free(density);
    free(density_II);
    free(current);
    free(blockStartsForSplitting);
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
 * @brief Checks that the hardcore constrains are not violated, throw error if conditions are violated
 * 
 */
void checkConfiguration(int id, std::string where){
    int pos;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        if (fabs(getMinimumImageDistance(x[(pos+1)%N], x[pos])) - sigma < -pow(10, -10)/dt) {
            std::cout<<where<<std::endl;
            printState();
            std::cout<<std::endl;
            printf("Distance violated: %d, %d \n", pos, (pos+1)%N);
            std::cout.precision(17);
            std::cout<<x[(pos+1)%N]<<", "<< x[pos]<<std::endl;
            std::cout<<fabs(getMinimumImageDistance(x[pos], x[(pos+1)%N]))- sigma<<std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Perform measurements of physical quantities
 * 
 * @param id of the block
 */
void doMeasurements(int id){
    int pos;

    //collect statistics for one-particle density
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        density[int((x[pos] - floor(x[pos])+id) * spatialSteps)]++;
    }

    //collect statistics for two-particle density at contact
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        if(identification[pos] > 0){
            if(fabs(getMinimumImageDistance(x[(pos-1+N)%N], x[pos])) - sigma < delta){
                    density_II[int((x[(pos-1+N)%N] - floor(x[(pos-1+N)%N])+id) * spatialSteps)]++;
                }
            if(identification[pos] > 1){
                i += identification[pos] - 1;
            }
        }
    }

    //collect statistics for particle current
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        current[id] += v[pos];
    }

    //collect statistics for MSD
    MSD[id] = 0;
    for(int i = 0; i < blockSize[id]; i++){
        pos = i + blockStart[id];
        MSD[id] += (x_noPB[pos] - x_initial[pos]) * (x_noPB[pos] - x_initial[pos]);
    }
}

/**
 * @brief Perform one step of the simulation
 * 
 */
void doStep(){
    //array containing all threads
    std::thread threads[numberOfBlocks];

    //calculate particle velocities in parallel
    int seed;
    for(int i = 0; i < numberOfBlocks; i++){
        //generate seed for RNG generator of each thread
        seed = SeedGenerator();
        threads[i] = std::thread(particleVelocities, i, seed);
    }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }

    //determine good blocks for splitting in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(startOfSplits, i);
    }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }
    
    //split clusters in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(splitClusters, i);
    }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }

    //calculate cluster velocities in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(clusterVelocities, i);
    }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }

    do{ 
        //detect all collisions in parallel
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i] = std::thread(updateCollisions, i);
        }
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i].join();
        }
        
        //merge colliding clusters in parallel
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i] = std::thread(merge, i);
        }
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i].join();
        }

        //perform merging of clusters across block borders &
        //repeat until no more collisions are detected
    }while(mergeCrossBlockBorders());

    //advance time and particle positions in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(advanceTime, i);
        }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }
   
    //apply periodic boundary conditions in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(applyPeriodicBoundaryConditions, i);
        }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }

    //check configuration in parallel
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i] = std::thread(checkConfiguration, i, "end of step");
        }
    for(int i = 0; i < numberOfBlocks; i++){
        threads[i].join();
    }
}

/**
 * @brief Simulate the system for the given duration while performing measurements
 * 
 */
void simulateSystem(){
    double time = 0.0;                              /*current time*/
    std::thread threads[numberOfBlocks];            /*array containing all threads*/

    //prepare output files
    boost::filesystem::create_directory("output/");
    boost::filesystem::create_directory(outputPrefix);
    FILE *MSDFile;
    std::string filename = outputPrefix+"/MSD";
    MSDFile = fopen(filename.c_str(), "w");
    double tMSD = 0;
    double totalMSD;

    //perform simulation
    while(time < simulationDuration){
        doStep();
        time += dt;

        //do measurements in parallel
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i] = std::thread(doMeasurements, i);
        }
        for(int i = 0; i < numberOfBlocks; i++){
            threads[i].join();
        }

        //calculate MSD
        tMSD += dt;
        if (tMSD >= tfMSD) {
            tMSD  = 0.0;
            tfMSD *= 1.1;

            totalMSD = 0;
            for(int i = 0; i < numberOfBlocks; i++){
                totalMSD += MSD[i];
            }

            fprintf(MSDFile, "%f, ", time);
            fprintf(MSDFile, "%f\n", totalMSD/N);
        }
    }
    //close output file
    fclose(MSDFile);
}

/**
 * @brief Simulate the system for the given duration without performing measurements
 * 
 */
void getEquilibriumState(){
    double time = 0.0;
    while(time < equilibrationTime){
        doStep();
        time += dt;
    }

    //reset starting positions for MSD calculation
    for(int i = 0; i<N; i++){
        x_initial[i] = x[i];
        x_noPB[i] = x[i];
    }
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
    FILE *Density_IIFile;
    std::string filename = outputPrefix+"/current";
    currentFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/density";
    DensityFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/density_II";
    Density_IIFile = fopen(filename.c_str(), "w");

    //get densities from all blocks
    for(int i = spatialSteps; i<spatialSteps*numberOfBlocks; i++){
        density[i%spatialSteps] += density[i];
        density_II[i%spatialSteps] += density_II[i];
    }

    //save one and two particle density profiles
    for(int i = 0; i < spatialSteps; ++i){
        fprintf(DensityFile, "%f, ", i * delta);
        fprintf(DensityFile, "%f\n", density[i]/(simulationDuration/dt*L*delta));
        fprintf(Density_IIFile, "%f, ", i * delta);
        fprintf(Density_IIFile, "%f\n", density_II[i]/(simulationDuration/dt*L*delta*deltaPrime));
    }

    //get particle current from all blocks
    for(int i = 1; i < numberOfBlocks; i++){
        current[0] += current[i];
    }

    //save particle current
    fprintf(currentFile, "%f", current[0]*dt/L/simulationDuration);
    
    //close output files
    fclose(currentFile);
    fclose(DensityFile);
    fclose(Density_IIFile);
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
    ("tCurrent", po::value<double>(&tfCurrent), "Time interval of outputting the current.")
    ("tTrajectory", po::value<double>(&tfTrajectory), "Time interval of outputting the particle positions.")
    ("tMSD", po::value<double>(&tfMSD), "Time interval of outputting the mean square displacement.")
    ("delta", po::value<double>(&delta), "Spatial resolution.")
    ("deltaPrime", po::value<double>(&deltaPrime), "Spatial resolution for particles in contact.")
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
    std::cout<<"current output frequency: "<<tfCurrent<<std::endl;
    std::cout<<"trajectory output frequency: "<<tfTrajectory<<std::endl;
    std::cout<<"MSD output frequency: "<<tfMSD<<std::endl;
    std::cout<<"delta = "<<delta<<std::endl;
    std::cout<<"deltaPrime = "<<deltaPrime<<std::endl;
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