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


#include <iostream>
#include <armadillo>
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
#include <chrono>

//generator for random numbers (normal distribution)
boost::random::mt19937 Generator1((std::chrono::system_clock::now().time_since_epoch()).count());
boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > randNormal(Generator1, boost::normal_distribution<>());

//generator for random numbers (uniform distribution [0, 1])
boost::random::mt19937 Generator2((std::chrono::system_clock::now().time_since_epoch()).count());
boost::variate_generator<boost::mt19937&, boost::uniform_01<> > randUniform(Generator2, boost::uniform_01<>());

arma::vec x;                                        /*particle positions*/                                            
arma::vec x_initial;                                /*initial particle positions*/
arma::vec x_noPB;                                   /*particle positions without periodic boundaries*/ 
arma::vec x_old;                                    /*old particle positions*/ 
arma::vec v;                                        /*particle velocities*/   
arma::vec configuration;                            /*cluster identification:   1 = independent particle
                                                                                0 = particle is inside of a cluster
                                                                                2, 3, .., N = first particle of a cluster with this length
                                                                                -2, -3, .., -N = last particle of a cluster with this length*/

//arma::vec numOfMerges;                              /*counter for the number of merges*/
//arma::vec numOfSplits;                              /*counter for the number of splits*/
//arma::vec timeOfMerges;                             /*total computation time of merges*/
//arma::vec timeOfSplits;                             /*total computation time of splits*/
//double meanClusterSize;                             /*mean cluster size*/
//arma::vec MergesAsFunctionOfMeanClusterSize;        /*number of merges as a function of the mean cluster size*/
//arma::vec numMergesAsFunctionOfMeanClusterSize;     /*how often does a certain mean cluster size occur after merging*/
bool save_all_jumps = false;                        /*write every jump across potential barrier to file*/

arma::vec density;                                  /*one-partcile density*/
arma::vec density_II;                               /*two-particle density at contact*/
int spatialSteps;                                   /*spatial resolution when measuring densities*/
double current;                                     /*particle current*/

double pi = boost::math::double_constants::pi;      /*pi*/
arma::vec deterministicForce;                       /*external force*/
arma::vec interactionForce;                         /*interaction force*/
int forceSteps = 1000000;                           /*spatial resolution for pre-calculated forces*/

double U0 = 6;                                      /*amplitude of cosine potentail*/
double D = 1;                                       /*diffusion coefficient*/
double sigma = 0.5;                                 /*particle size*/
int L = 100;                                        /*system length*/
int N = 50;                                         /*number of particles*/
double Gamma = 1.0;                                 /*amplitude of adhesive interaction*/
int order = 3;                                      /*order of polynomial representation of delta function*/
double epsilon = 0.05;                              /*range of adhesive interaction*/
double dt = 0.001;                                  /*time step*/
double f = 0;                                       /*constant drag force*/
double delta = 0.001;                               /*spatial resolition for measuring denity and two-particle-density at contact*/
double deltaPrime = delta;                          /*spatial resolition for measuring two-particle-density at contact*/
double equilibrationTime = 1000.0;                  /*time to run th esimulation before starting measurements*/
double totalSimulationTime = 3000.0;                /*total simulation time*/
double tfTrajectory = 0.001*totalSimulationTime;    /*time between outputting the particle positions*/
double tfCurrent = 0.001*totalSimulationTime;       /*time between outputting the particle current*/
double tfMSD = 0.001;                               /*time between outputting the mean square displacement*/
std::string outputPrefix = "";                      /*prefix of output (folder name)*/
std::string cfgFile = "basep.cfg";                  /*default file name of configuration parameters*/

//double numberOfSplits;                              /*count the number of splits*/
//double numberOfCollisions;                          /*count number of collisions*/
//long numberOfContacts;                              /*count number of contact between particles*/

/**
 * @brief Maps the particle position back into the central image box according to the boundary conditions.
 * 
 */
void applyPeriodicBoundaryConditions(){
    for(int i = 0; i < N; ++i){
        // Apply periodic boundary conditions
        if(x(i)>= L){
            x(i) -= L;
        }else if(x(i) < 0.0){
            x(i) += L;
        }
    }
}

/**
 * @brief Calculate the instantaneous mean cluster size
 * 
 * @return double 
 */
double getMeanClusterSize(){
    double num = 0;
    double size = 0;
    for(int i = 0; i < N; i++){
        if(configuration(i)>0){
            size += configuration(i);
            num += 1;
        }
    }
    return size/num;
}

/**
 * @brief Setting random initial particle positions, initializing pre-calculated forces
 * 
 */
void initialConfiguration(){
    spatialSteps = int(round(1/delta));             
    x = arma::zeros<arma::vec>(N);
    x_initial = arma::zeros<arma::vec>(N);
    x_noPB = arma::zeros<arma::vec>(N);
    v = arma::zeros<arma::vec>(N);
    current = 0;

    //numOfMerges = arma::zeros<arma::vec>(N);
    //numOfSplits = arma::zeros<arma::vec>(100*N);
    //timeOfMerges = arma::zeros<arma::vec>(N);
    //timeOfSplits = arma::zeros<arma::vec>(100*N);
    //meanClusterSize  = 0;
    //MergesAsFunctionOfMeanClusterSize = arma::zeros<arma::vec>(100*N);
    //numMergesAsFunctionOfMeanClusterSize = arma::zeros<arma::vec>(100*N);

    //initalize configuration: all particles are independent
    configuration = arma::ones<arma::vec>(N);

    /*Windows for each particle position are calculated so that there are no conflicts.
    Particle positions are put in each window according to a uniform distribution.*/
    for (int i = 0; i < N; ++i) {
        x(i) = double(L)/double(N) * i + sigma/2 + randUniform() * (double(L)/double(N)-sigma);
    }

    x_initial = x;
    x_noPB = x;

    //initialize one/two-particle densities
    density = arma::zeros<arma::vec>(spatialSteps);
    density_II = arma::zeros<arma::vec>(spatialSteps);

    interactionForce = arma::zeros<arma::vec>(forceSteps);
    deterministicForce = arma::zeros<arma::vec>(forceSteps);

    //pre-calculate interaction forces
    for(int i = 0; i < forceSteps; i++){
        deterministicForce(i) = pi * U0 * sin(2 * pi * double(i)/double(forceSteps)) + f;
    }

    //pre-calculate interaction forces
    double distance;
    for(int i = 0; i < forceSteps; i++){
        distance = double(i)/double(forceSteps) + sigma;

        if(distance >= sigma + epsilon){
            interactionForce(i) = 0.0;
        }else{
            interactionForce(i) = -double(order)*Gamma*((pow(epsilon+sigma-distance, order-1))/((pow(epsilon, order+1)/(order+1))+Gamma*pow(epsilon+sigma-distance, order)));
        }
    }

    //numberOfSplits = 0;
    //numberOfCollisions = 0;
    //numberOfContacts = 0;
}

/**
 * @brief Get the minimum image distance between particles
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
void particleVelocities(){
    double interaction = 0;
    double distance;

    for(int i = 0; i < N; i++){
        interaction = 0;
        //get interaction forces, if particles are within interaction range epsilon
        distance = getMinimumImageDistance(x(i), x((i-1+N)%N));
        if(distance < sigma + epsilon){
            interaction += interactionForce(int(round((distance-sigma)*forceSteps)));
        }
        distance = getMinimumImageDistance(x((i+1)%N), x(i));
        if(distance < sigma + epsilon){
            interaction -= interactionForce(int(round((distance-sigma)*forceSteps)));
        }
        //calculate particle velocity
        v(i) = deterministicForce((x(i) - floor(x(i))) * forceSteps) + interaction + sqrt(2*D/dt)*randNormal();
    }
}

/**
 * @brief Calculate cluster velocities from culster identification and particle velocities
 * 
 */
void clusterVelocities(){
    for(int i = 0; i < N; i++){
        if(configuration(i)>0){
            for(int j = 1; j < configuration(i); j++){
                v(i) += v((i+j)%N);
            }
            v(i) /= configuration(i);

            for(int j = 1; j < configuration(i); j++){
                v((i+j)%N) = v(i);
            }
        }
    }
}

/**
 * @brief Splitting clusters at the correct positions
 * 
 */
void splitClusters(){ 
    double forceLeft;                               /*total force on the left subcluster*/
    double forceRight;                              /*total force on the right subcluster*/

    int relativePosOfBiggestDifference;             /*position of potential split*/
    double lowestInteractionForce;                  /*lowest difference of the interaction forces of the two subclusters found so far*/

    int firstIndependentCluster = -1;               /*index of the first cluster*/
    int totalChecks = N;                            /*total number of particles to check*/
    int i;                                          /*index of the current particle*/

    //check all clusters
    for(int n = 0; n < totalChecks; n++){
        i = n%N;
        //find next cluster
        if(configuration(i)>1){
            //find the first particle of the first cluster
            if(firstIndependentCluster < 0){
                firstIndependentCluster = i;
                totalChecks = i + N;
            }

            forceLeft = 0;
            forceRight = 0;

            relativePosOfBiggestDifference = -1;
            lowestInteractionForce = 0;

            //calculate total force exerted on the cluster
            for(int j = 0; j < configuration(i); j++){
                forceRight += v((i+j)%N);
            }

            //calculate forces on two subclusters for all possible splitting positions
            for(int j = 0; j < configuration(i)-1; j++){ /*added -1 to fix this (#connections = #particles - 1)*/
                //update forces of both subclusters
                forceRight -= v((i+j)%N);
                forceLeft += v((i+j)%N);

                //find the pair of subclusters for which the velocity difference is the largest
                if(forceLeft/(j+1) - forceRight/(configuration(i)-j-1) < lowestInteractionForce){
                    lowestInteractionForce = forceLeft/(j+1) - forceRight/(configuration(i)-j-1);
                    relativePosOfBiggestDifference = j;
                }      
            }

            //if the difference is positive, we split the cluster by updateing the configuration
            if(relativePosOfBiggestDifference > -1){
                configuration((i+relativePosOfBiggestDifference+1)%N) = configuration(i) - relativePosOfBiggestDifference - 1;
                configuration(i) = relativePosOfBiggestDifference + 1; 

                //check the new cluster again
                n--;

                if(configuration(i) > 1){
                    configuration((i+int(configuration(i))-1)%N) = - configuration(i);
                }

                if(configuration((i+relativePosOfBiggestDifference+1)%N) > 1){
                    configuration((i+relativePosOfBiggestDifference+1+int(configuration((i+relativePosOfBiggestDifference+1)%N))-1)%N) = - configuration((i+relativePosOfBiggestDifference+1)%N);
                }
                //numberOfSplits += 1;
            } 
        }
    }
}


/**
 * @brief Checks that the hardcore constrains are not violated, throw error if conditions are violated
 * 
 */
void checkConfiguration(){
    for (int i = 0; i < N-1; ++i) 
        if (fabs(getMinimumImageDistance(x(i+1), x(i))) - sigma < -pow(10, -12)/dt) {
            printf("Distance violated: %d, %d \n", i, i+1);
            std::cout.precision(17);
            std::cout<<x(i+1)<<", "<< x(i)<<std::endl;
            std::cout<<fabs(getMinimumImageDistance(x(i), x(i+1)))- sigma<<std::endl;
            exit(EXIT_FAILURE);
        }
		if (fabs(getMinimumImageDistance(x(N-1), x(0))) - sigma < -pow(10, -12)/dt) {
            printf("Distance violated: %d, %d \n", N-1, 0);
            std::cout.precision(17);
            std::cout<<x(N-1)<<", "<< x(0)<<std::endl;
            std::cout<<fabs(getMinimumImageDistance(x(N-1), x(0)))- sigma<<std::endl;
            exit(EXIT_FAILURE);
    }
}

/**
 * @brief Total sum of all positions of the particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @return double 
 */
double getClusterCM(int id){
    id += 2*N;
    id = id%N;
    return x(id) * configuration(id) + sigma * ((configuration(id)-1) * configuration(id)) / 2;
}

/**
 * @brief Total sum of all velocities of the particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @return double 
 */
double getClusterV(int id){
    id += 2*N;
    id = id%N;
    return v(id) * configuration(id);
}

/**
 * @brief Number of particles in the cluster
 * 
 * @param id of the first particle of the cluster
 * @return double 
 */
double getClusterSize(int id){
    id += 2*N;
    id = id%N;
    return int(configuration(id));
}

/**
 * @brief Get the id of the next cluster to the left
 * 
 * @param id of the first particle of the cluster
 * @return int 
 */
int getLeftId(int id){
    if(configuration((id-1+N)%N)<0){
        return int(configuration((id-1+N)%N)) + id;
    }else{
        return id - 1;
    }
}

/**
 * @brief Execute a single time step of duration dt
 * 
 */
void doSingleTimeStep(){ 
    //remember old particle positions
    x_old = x;

    //double currentClusterSize = getMeanClusterSize();
    //meanClusterSize += currentClusterSize * dt;
    //int countMerges = 0;

    //calculate new particle velocities
    particleVelocities();

    //auto start = std::chrono::system_clock::now();

    //cluster analysis (fragmentation)
    splitClusters();

    //auto end = std::chrono::system_clock::now();
    //std::chrono::duration<double> elapsed_seconds = end-start;
    //timeOfSplits(round(100*currentClusterSize)-1) += elapsed_seconds.count();
    //numOfSplits(round(100*currentClusterSize)-1) += 1;

    //calculate cluster velocities based on forces exerted on particles and cluster identification
    clusterVelocities();
    
    //start = std::chrono::system_clock::now();

    //keep 3 neighboring clusters in memory at all times
    int thisId, thisSize;
    double thisCM, thisV;

    int leftId, leftSize;
    double leftCM, leftV;

    int rightId, rightSize;
    double rightCM, rightV;

    double dx, dv;
    int collisionHappened = 0;
    int firstId;

    //find first cluster
    thisId = 0;                                        
    while(configuration(thisId)<1){
        thisId++;
    }
    firstId = thisId;

    //load data for all 3 clusters
    thisCM = getClusterCM(thisId);
    thisV = getClusterV(thisId);
    thisSize = getClusterSize(thisId);

    leftId = getLeftId(thisId);
    rightId = int(configuration(thisId%N)) + thisId;

    leftCM = getClusterCM(leftId);
    leftV = getClusterV(leftId);
    leftSize = getClusterSize(leftId);

    rightCM = getClusterCM(rightId);
    rightV = getClusterV(rightId);
    rightSize = getClusterSize(rightId);
    do{
        //calculate time until next collision
        dv = thisV/thisSize - rightV/rightSize;
        dx = rightCM/rightSize - sigma * double(rightSize+thisSize)/2 - thisCM/thisSize;
        if(rightCM/rightSize < sigma * double(rightSize+thisSize-2)/2 + thisCM/thisSize){
            dx += L;
        }
        
        //if collision happens within time dt, merge clusters
        if(dv > 0 && dt >= dx/dv){
            if(thisCM/thisSize > rightCM/rightSize){
                rightCM += rightSize * L;
            }
            rightCM +=thisCM;
            rightV += thisV;
            rightSize += thisSize;
            rightId = thisId;

            thisId = leftId;
            thisV = leftV;
            thisCM = leftCM;
            thisSize = leftSize;

            //load data for the new cluster neighboring to the left
            leftId = getLeftId(thisId);
            leftCM = getClusterCM(leftId);
            leftV = getClusterV(leftId);
            leftSize = getClusterSize(leftId);

            //remember that  collision happened
            collisionHappened = 1;

            //numberOfCollisions += 1;
            //countMerges += 1;
        }else{
            //if a merge happened but no further merges are detected for the same cluster at this time, update the configuration
            if(collisionHappened > 0){
                for(int i = 0; i<rightSize; i++){
                    configuration((i+rightId+N)%N) = 0;
                    v((i+rightId+N)%N) = rightV/rightSize;
                    x((i+rightId+N)%N) = rightCM/rightSize + sigma * double(2*i-rightSize+1)/2;
                }
                configuration((rightId+N)%N) = rightSize;
                configuration((rightId+rightSize-1+N)%N) = -rightSize;
                collisionHappened = 0;
            }
            
            //move on to the next cluster
            leftId = thisId;
            leftCM = thisCM;
            leftSize = thisSize;
            leftV = thisV;

            thisId = rightId;
            thisV = rightV;
            thisCM = rightCM;
            thisSize = rightSize;
            
            //load data for the new cluster neighboring to the right
            rightId = int(configuration((thisId+N)%N)) + thisId;
            
            rightCM = getClusterCM(rightId);
            
            rightV = getClusterV(rightId);
         
            rightSize = getClusterSize(rightId);
        }
        //do 1 full cycle through the system
    }while(firstId + N >= thisId && thisId >= firstId - N);

    //end = std::chrono::system_clock::now();
    //elapsed_seconds = end-start;
    //timeOfMerges(countMerges) += elapsed_seconds.count();
    //numOfMerges(countMerges) += 1;
    //MergesAsFunctionOfMeanClusterSize(round(100*currentClusterSize)-1) += countMerges;
    //numMergesAsFunctionOfMeanClusterSize(round(100*currentClusterSize)-1) += 1;

    //move partciles according to velocities
    x += dt * v;

    for(int i = 0; i<N; i++){
        if(x(i)-x_old(i)>L/2){
            x_old(i)+=L;
        }
        if(x(i)-x_old(i)<-L/2){
            x_old(i)-=L;
        }
        //track particle positions without periodic boundaries for MSD calculation
        x_noPB(i) += x(i)-x_old(i); 
    }

    applyPeriodicBoundaryConditions();
    checkConfiguration();

    //measure current
    current += arma::accu(v)*dt/L;
}

/**
 * @brief Performing the simulation for the equilibration time without output
 * 
 */
void getEquilibriumState(){
    double timePassed = 0;                              /*time passed since start of simulation*/
    //run simulation for the equilibration time
    while(timePassed < equilibrationTime){
        doSingleTimeStep();
        timePassed += dt;
    }

    //reset output variables
    current = 0;
    //numberOfSplits = 0;
    //numberOfCollisions = 0;
    //numberOfContacts = 0;

    x_initial = x;
    x_noPB = x;

    //numOfMerges = arma::zeros<arma::vec>(N);
    //numOfSplits = arma::zeros<arma::vec>(100*N);
    //timeOfMerges = arma::zeros<arma::vec>(N);
    //timeOfSplits = arma::zeros<arma::vec>(100*N);
    //meanClusterSize  = 0;
    //MergesAsFunctionOfMeanClusterSize = arma::zeros<arma::vec>(100*N);
    //numMergesAsFunctionOfMeanClusterSize = arma::zeros<arma::vec>(100*N);
}

/**
 * @brief Run the Brownian dynamics simulation
 * 
 */
void simulateSystem(){
    double timePassed = 0;                              /*time passed since start of simulation*/
    double tTrajectory = 0;                             /*time passed since last output of trajectory*/
    double tCurrent = 0;                                /*time passed since last output of current*/
    double tMSD = 0;                                    /*time passed since last output of MSD*/
    double MSD;                                         /*mean square displacement*/
    int value;                                          /*temporary variable for writing to file*/
    bool forward = true;                                /*direction of jump*/
    bool backward = false;                              /*direction of jump*/

    //prepare output files
    boost::filesystem::create_directory("output/");
    boost::filesystem::create_directories(outputPrefix);
    FILE *trajectoryFile;
    FILE *currentFile;
    FILE *MSDFile;
    std::string filename = outputPrefix+"/trajectory";
    trajectoryFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/current";
    currentFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/MSD";
    MSDFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/jump_particle.bin";
    std::ofstream file_jump_particle(filename, std::ios::binary);
    filename = outputPrefix+"/jump_time.bin";
    std::ofstream file_jump_time(filename, std::ios::binary);
    filename = outputPrefix+"/jump_direction.bin";
    std::ofstream file_jump_direction(filename, std::ios::binary);

    //run simulation for the total simulation time
    while(timePassed < totalSimulationTime){
        doSingleTimeStep();
        //write jumps across potential barrier to file
        if(save_all_jumps){
            for(int i = 0; i < N; i++){
                if(int(x(i))-int(x_old(i)) == 1){
                    value = int(x_old(i));
                    file_jump_particle.write(reinterpret_cast<char*>(&value), sizeof(value));
                    file_jump_time.write(reinterpret_cast<char*>(&timePassed), sizeof(timePassed));
                    file_jump_direction.write(reinterpret_cast<char*>(&forward), sizeof(forward));
                }
                if(int(x(i))-int(x_old(i)) == -1){
                    value = int(x(i));
                    file_jump_particle.write(reinterpret_cast<char*>(&value), sizeof(value));
                    file_jump_time.write(reinterpret_cast<char*>(&timePassed), sizeof(timePassed));
                    file_jump_direction.write(reinterpret_cast<char*>(&backward), sizeof(backward));
                }
            }
            
        }
        //update time
        timePassed += dt;

        //collect statistics for one-particle density
        for(int i = 0; i < N; i++){
            density((x(i) - floor(x(i))) * spatialSteps)++;
        }

        //collect statistics for two-particle density at contact
        for(int i = 0; i < N; i++){
            if(configuration(i) > 0){
                if(fabs(getMinimumImageDistance(x((i-1+N)%N), x(i))) - sigma < delta){
                        density_II((x((i-1+N)%N) - floor(x((i-1+N)%N))) * spatialSteps)++;
                    }
                if(configuration(i) > 1){
                    i += configuration(i) - 1;
                }
            }
        }

        //output trajectory at each time interval tfTrajectory
        tTrajectory += dt;
        if (tTrajectory >= tfTrajectory) {
            tTrajectory = 0.0;
            fprintf(trajectoryFile, "%f, ", timePassed);
            for(int i = 0; i < N-1; ++i){
                fprintf(trajectoryFile, "%f, ", x(i));
            }
            fprintf(trajectoryFile, "%f\n", x(N-1));
        }

        //output current at each time interval tfCurrent
        tCurrent += dt;
        if (tCurrent >= tfCurrent) {
            tCurrent  = 0.0;
            fprintf(currentFile, "%f, ", timePassed);
            fprintf(currentFile, "%f\n", current/timePassed);
        }

        //calculate mean square displacement
        MSD = 0;
        for(int i = 0; i < N; i++){
            MSD += (x_noPB(i) - x_initial(i)) * (x_noPB(i) - x_initial(i));
        }

        //output MSD
        tMSD += dt;
        if (tMSD >= tfMSD) {
            tMSD  = 0.0;
            tfMSD *= 1.1;
            fprintf(MSDFile, "%f, ", timePassed);
            fprintf(MSDFile, "%f\n", MSD/N);
        }

        //update number of contacts between particles
        //for(int i = 0; i < N; i++){
        //    if(configuration(i)==0 || configuration(i)>1){
        //        numberOfContacts++;
        //    }
        //}
    }

    //close output files
    fclose(trajectoryFile);
    fclose(currentFile);
    fclose(MSDFile);
    file_jump_particle.close();
    file_jump_time.close();
    file_jump_direction.close();

    //normalize densities
    density /= totalSimulationTime/dt*L*delta;
    density_II /= totalSimulationTime/dt*L*delta*deltaPrime;
    //timeOfMerges /= numOfMerges;
    //timeOfSplits /= numOfSplits;
    //MergesAsFunctionOfMeanClusterSize /= numMergesAsFunctionOfMeanClusterSize;
}

/**
 * @brief Print densities to file
 * 
 */
void saveResults(){
    arma::mat output = arma::zeros<arma::mat>(spatialSteps, 2);
    for(int i = 0; i < spatialSteps; i++){
        output(i, 0) = i * delta;
        output(i, 1) = density(i);
    }
    output.save(outputPrefix+"/density", arma::raw_ascii);

    /*for(int i = 0; i < spatialSteps; i++){
        output(i, 1) = density_II(i);
    }
    output.save(outputPrefix+"/density_II", arma::raw_ascii);

    output = arma::zeros<arma::mat>(100*N, 2);
    for(int i = 0; i < 100*N; i++){
        output(i, 0) = double(i) / double(100);
        output(i, 1) = timeOfSplits(i);
    }
    output.save(outputPrefix+"/timeOfSplits", arma::raw_ascii);

    output = arma::zeros<arma::mat>(N, 2);
    for(int i = 0; i < N; i++){
        output(i, 0) = i;
        output(i, 1) = timeOfMerges(i);
    }
    output.save(outputPrefix+"/timeOfMerges", arma::raw_ascii);

    output = arma::zeros<arma::mat>(100*N, 2);
    for(int i = 0; i < 100*N; i++){
        output(i, 0) = double(i) / double(100);
        output(i, 1) = MergesAsFunctionOfMeanClusterSize(i);
    }
    output.save(outputPrefix+"/MergesAsFunctionOfMeanClusterSize", arma::raw_ascii);

    arma::vec splitsAndCollisions = arma::zeros<arma::vec>(2);
    splitsAndCollisions(0) = numberOfSplits/totalSimulationTime*dt;
    splitsAndCollisions(1) = numberOfCollisions/totalSimulationTime*dt;
    splitsAndCollisions.save(outputPrefix+"/splitsAndCollsiions", arma::raw_ascii);

    arma::vec meanCS = arma::zeros<arma::vec>(1);
    meanCS(0) = meanClusterSize/totalSimulationTime;
    meanCS.save(outputPrefix+"/meanClusterSize", arma::raw_ascii);

    arma::vec contacts = arma::zeros<arma::vec>(1);
    contacts(0) = double(numberOfContacts)/totalSimulationTime*dt;
    contacts.save(outputPrefix+"/numberOfContacts", arma::raw_ascii);*/
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
    ("totaltime,t", po::value<double>(&totalSimulationTime), "Total time of simulation.")
    ("particles,N", po::value<int>(&N), "Number of particles.")
    ("length,L", po::value<int>(&L), "Length of the simulated system.")
    ("diameter,sigma", po::value<double>(&sigma), "Diameter of hard rods.")
    ("DiffusionConstant,D", po::value<double>(&D), "Diffusion constant.")
    ("timestep,dt", po::value<double>(&dt), "Timestep for Langevin-Euler-Scheme.")
    ("force,f", po::value<double>(&f), "Amplitude of constants drag force.")
    ("AmpU,U", po::value<double>(&U0), "Amplitude of external periodic potential.")
    ("gamma,stickyness", po::value<double>(&Gamma), "Adhesive interaction strength.")
    ("order", po::value<int>(&order), "Polynomial order of the representation of the delta function.")
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

    std::cout<<"totaltime: "<<totalSimulationTime<<std::endl;
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
 * pure hard spheres (gamma = 0) in an external cosine potential
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
    initialConfiguration();

    //run simulation until equilibrium is reached
    getEquilibriumState();

    //save start time
    auto start = std::chrono::system_clock::now();

    //run the simulation
    simulateSystem();

    //save end time
    auto end = std::chrono::system_clock::now();

    //calculate the computation time
    std::chrono::duration<double> elapsed_seconds = end-start;

    //print computation time
    std::cout<<"Computation time: "<<elapsed_seconds.count()<<" s"<<std::endl;

    //save results
    saveResults();
}