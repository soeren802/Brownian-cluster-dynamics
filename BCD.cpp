/*
The MIT License

Copyright © 2022 Alexander P. Antonov, Sören Schweers, Artem Ryabov, and Philipp Maass

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

#include <armadillo>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/filesystem.hpp>

//generator for random numbers (normal distribution)
boost::random::mt19937 Generator1((std::chrono::system_clock::now().time_since_epoch()).count());
boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > randNormal(Generator1, boost::normal_distribution<>());

//generator for random numbers (uniform distribution [0, 1])
boost::random::mt19937 Generator2((std::chrono::system_clock::now().time_since_epoch()).count());
boost::variate_generator<boost::mt19937&, boost::uniform_01<> > randUniform(Generator2, boost::uniform_01<>());

arma::vec x;                                        /*particle positions*/  
arma::vec x_initial;                                /*initial particle positions*/
arma::vec x_noPB;                                   /*particle positions without periodic boundaries*/                                         
arma::vec v;                                        /*particle velocities*/    
arma::vec configuration;                            /*cluster identification: 1 = independent particle; 0 = particle is part of a cluster; 2,3,..,N = first particle of a cluster with this length*/

arma::vec numOfMerges;
arma::vec timeOfMerges;
double meanClusterSize;

arma::vec density;                                  /*one-partcile density*/
arma::vec density_II;                               /*two-particle density*/
int spatialSteps;                                   /*spatial resolution when measuring densities*/
double current;                                     /*particle current*/
double MSD = 0;                                     /*mean square displacement*/

double pi = boost::math::double_constants::pi;      /*pi*/
arma::vec deterministicForce;                       /*external force*/
arma::vec interactionForce;                         /*interaction force*/
int forceSteps = 1000000;                           /*spatial resolution for pre-calculated forces*/

double U0 = 6;                                      /*amplitude of cosine potentail*/
double D = 1;                                       /*diffusion coefficient*/
double sigma = 0.5;                                 /*particle size*/
int L = 100;                                        /*system length*/
int N = 50;                                         /*particles number*/
double Gamma = 1.0;                                 /*amplitude of adhesive interaction*/
int order = 3;                                      /*order of polynomial representation of delta function*/
double epsilon = 0.05;                              /*range of adhesive interaction*/
double dt = 0.0000001;                              /*time step*/
double f = 0;                                       /*constant drag force*/
double delta = 0.001;                               /*spatial resolition for measuring denity and two-particle-density at contact*/
double deltaPrime = delta;                          /*spatial resolition for measuring two-particle-density at contact*/
double equilibrationTime = 1000.0;                  /*time to run th esimulation before starting measurements*/
double totalSimulationTime = 4000.0;                /*total simulation time*/
double tfTrajectory = 0.001*totalSimulationTime;    /*time between outputting the particle positions*/
double tfCurrent = 0.001*totalSimulationTime;       /*time between outputting the particle current*/
double tfMSD = 0.001;                               /*time between outputting the mean square displacement*/
std::string outputPrefix = "";                      /*prefix of output (folder name)*/
std::string cfgFile = "basep.cfg";                  /*file name of configuration parameters*/

arma::vec taggedSingleParticlesEpsilon;             /*particles that have a distance larger than epsilon from other partilces at the beginning of the measurement*/
int numTaggedEpsilon;                               /*number of such partilces*/
arma::vec taggedSingleParticlesContact;             /*particles that not in contact with other partilces at the beginning of the measurement*/
int numTaggedContact;                               /*number of such partilces*/


/**
 * @brief Maps the particle position back into the central image box according to the boundary conditions. Updates the particle current accordingly. 
 * 
 */
void applyPeriodicBoundaryConditions(){
    for(int i = 0; i < N; ++i){
        if(x(i)>= L){
            x(i) -= L;
            current++;
        }else if(x(i) < 0.0){
            x(i) += L;
            current--;
        }
    }
}

/**
 * @brief Get the Mean Cluster Size object
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
    x = arma::zeros<arma::vec>(N);
    x_initial = arma::zeros<arma::vec>(N);
    x_noPB = arma::zeros<arma::vec>(N);
    v = arma::zeros<arma::vec>(N);
    current = 0;

    numOfMerges = arma::zeros<arma::vec>(N);
    timeOfMerges = arma::zeros<arma::vec>(N);
    meanClusterSize  = 0;

    //initalize configuration: all particles are independent
    configuration = arma::ones<arma::vec>(N);
    
    /*windows for each particle position are calculated so that there are no conflicts,
    particle positions are put in each window according to a uniform distribution*/
    for (int i = 0; i < N; ++i) {
        x(i) = double(L)/double(N) * i + sigma/2 + randUniform() * (double(L)/double(N)-sigma);
    }

    x_initial = x;
    x_noPB = x;

    //initialize one/two-particle densities
    spatialSteps = int(round(1/delta));
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
}

/**
 * @brief Get the minimum image distance
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
 * @return arma::vec 
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
 * @brief Calculate cluster velocities from culster identification and forces exterted on each particle
 * 
 * @return arma::vec 
 */
void clusterVelocities(){
    for(int i = 0; i < N; i++){
        if(configuration(i)>0){
            //calculate cluster velocity
            for(int j = 1; j < configuration(i); j++){
                v(i) += v((i+j)%N);
            }
            v(i) /= configuration(i);

            //set cluster velocity as velocity for all particles in the cluster
            for(int j = 1; j < configuration(i); j++){
                v((i+j)%N) = v(i);
            }
        }
    }
}

/**
 * @brief Merge collided clusters into one cluster
 * 
 */
void mergeClusters(int collidingParticle){
    //find first particle of the left cluster
    int i = 0;
    while(configuration((collidingParticle-i+N)%N) < 1){
        i++;
    }
    
    v((collidingParticle-i+N)%N) = v((collidingParticle-i+N)%N)*configuration((collidingParticle-i+N)%N) + v((collidingParticle+1)%N)*configuration((collidingParticle+1)%N);

    //update cluster identification
    configuration((collidingParticle-i+N)%N) += configuration((collidingParticle+1)%N);
    configuration((collidingParticle+1)%N) = 0;

    //update velocities of all particles involved in the collision
    v((collidingParticle-i+N)%N) /= configuration((collidingParticle-i+N)%N);
    for(int j = 1; j < configuration((collidingParticle-i+N)%N); j++){
        v((collidingParticle-i+N+j)%N) = v((collidingParticle-i+N)%N);
    }
}

/**
 * @brief Splitting clusters at the correct positions
 * 
 */
void splitClusters(){ 
    double forceLeft;
    double forceRight;

    int relativePosOfBiggestDifference;
    double lowestInteractionForce;

    int firstIndependentCluster = -1;
    int totalChecks = N;
    int i;

    //check all clusters
    for(int n = 0; n < totalChecks; n++){
        i = n%N;
        if(configuration(i)>1){
            //get the first independent particle (i.e. single particle or first particle of a cluster)
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
            for(int j = 0; j < configuration(i)-1; j++){
                forceRight -= v((i+j)%N);
                forceLeft += v((i+j)%N);

                //find the pair of subclusters for which the velocity difference is the largest
                if(forceLeft/(j+1) - forceRight/(configuration(i)-j-1) < lowestInteractionForce){
                    lowestInteractionForce = forceLeft/(j+1) - forceRight/(configuration(i)-j-1);
                    relativePosOfBiggestDifference = j;
                }      
            }

	    //if the difference is positive, we split the cluster
            if(relativePosOfBiggestDifference > -1){
                configuration((i+relativePosOfBiggestDifference+1)%N) = configuration(i) - relativePosOfBiggestDifference - 1;
                configuration(i) = relativePosOfBiggestDifference + 1;  
                n--;
            } 
        }
    }
}

/**
 * @brief Checks if the hardcore constrains are not violated, throw error if conditions are violated
 * 
 */
void checkConfiguration(int collidingParticle){
    for (int i = 0; i < N-1; ++i) 
        if (fabs(getMinimumImageDistance(x(i+1), x(i))) - sigma < -pow(10, -12)/dt){
            printf("Distance violated: %d, %d \n", i, i+1);
            std::cout<<"collidingParticle = "<<collidingParticle<<std::endl;
            std::cout.precision(17);
            std::cout<<x(i+1)<<", "<< x(i)<<std::endl;
            std::cout<<fabs(getMinimumImageDistance(x(i), x(i+1)))- sigma<<std::endl;
            std::cout<<x<<std::endl;
            std::cout<<v<<std::endl;
            std::cout<<configuration<<std::endl;
            exit(EXIT_FAILURE);
        }
		if (fabs(getMinimumImageDistance(x(N-1), x(0))) - sigma < -pow(10, -12)/dt){
            printf("Distance violated: %d, %d \n", N-1, 0);
            std::cout<<"collidingParticle = "<<collidingParticle<<std::endl;
            std::cout.precision(17);
            std::cout<<x(N-1)<<", "<< x(0)<<std::endl;
            std::cout<<fabs(getMinimumImageDistance(x(N-1), x(0)))- sigma<<std::endl;
            std::cout<<x<<std::endl;
            std::cout<<v<<std::endl;
            std::cout<<configuration<<std::endl;
            exit(EXIT_FAILURE);
    }
    if(arma::accu(configuration) != N){
        std::cout<<"Error in configuration"<<std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Execute a single time step of length dt
 * 
 */
void doSingleTimeStep(){
    meanClusterSize += getMeanClusterSize() * dt;
    double firstCollision;
    double remainingTime = dt;
    double dx, dv;
    int collidingParticle;

    //calculate new particle velocities
    particleVelocities();

    //cluster analysis (fragmentation)
    splitClusters();

    //calculate cluster velocities based on forces exerted on particles and cluster identification
    clusterVelocities();

    int countMerges = 0;
    auto start = std::chrono::system_clock::now();
    while(remainingTime > 0){
        //calculate time of first collision
        firstCollision = remainingTime;
        collidingParticle = -1;
        for(int i = 0; i < N; i++){
            //calculate distance between particles
            dx = x(i) - x((i-1+N)%N) - sigma;
            //apply periodic boundaries
            if(x((i-1+N)%N) > x(i)){
                dx += L;
            }
            dv = v((i-1+N)%N) - v(i);
            //check whether particles are moving towards each other
            if(dv > 0){
                //calculate lowest collision time
                if(firstCollision > dx/dv){
                    firstCollision = dx/dv;
                    collidingParticle = (i-1+N)%N;
                }
            }
        }

        //move partciles according to velocities
        x += firstCollision * v;
        x_noPB += firstCollision * v;
        applyPeriodicBoundaryConditions();

        //record time passed
        remainingTime -= firstCollision;

        //check for particle overlaps
        checkConfiguration(collidingParticle);

        //merge collided clusters
        if(collidingParticle > -1){
            mergeClusters(collidingParticle);
            countMerges += 1;
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    timeOfMerges(countMerges) += elapsed_seconds.count();
    numOfMerges(countMerges) += 1;
}

/**
 * @brief Performing the simulation for the equilibration time without output
 * 
 */
void getEquilibriumState(){
    double timePassed = 0;
    while(timePassed < equilibrationTime){
        doSingleTimeStep();
        timePassed += dt;
    }

    //resetting current
    current = 0;

    //resetting MSD
    MSD = 0;
    x_initial = x;
    x_noPB = x;

    numOfMerges = arma::zeros<arma::vec>(N);
    timeOfMerges = arma::zeros<arma::vec>(N);
    meanClusterSize  = 0;
}

/**
 * @brief tag single particles for MSD measurement initially sepearte particles
 * 
 */
void tagParticles(){
    numTaggedEpsilon = 0;
    numTaggedContact = 0;
    arma::vec taggedEpsilonTemp = arma::zeros<arma::vec>(N);
    arma::vec taggedContactTemp = arma::zeros<arma::vec>(N);

    for(int i = 0; i<N; i++){
        if(getMinimumImageDistance(x((i+1)%N), x(i)) - sigma > epsilon && getMinimumImageDistance(x(i), x((i-1+N)%N)) - sigma > epsilon){
            taggedEpsilonTemp(numTaggedEpsilon) = i;
            numTaggedEpsilon++;
        }
        if(configuration(i) == 1){
            taggedContactTemp(numTaggedContact) = i;
            numTaggedContact++;
        }
    }

    taggedSingleParticlesEpsilon = arma::zeros<arma::vec>(numTaggedEpsilon);
    taggedSingleParticlesContact = arma::zeros<arma::vec>(numTaggedContact);

    for(int i = 0; i<numTaggedEpsilon; i++){
       taggedSingleParticlesEpsilon(i) = taggedEpsilonTemp(i);
    }
    for(int i = 0; i<numTaggedContact; i++){
       taggedSingleParticlesContact(i) = taggedContactTemp(i);
    }
}

/**
 * @brief Run the Brownian dynamics simulation
 * 
 */
void simulateSystem(){
    double timePassed = 0;
    double tTrajectory = 0;
    double tCurrent = 0;
    double tMSD = 0;
    int k;

    //tag particles for MSD measurement
    tagParticles();

    //prepare output files
    boost::filesystem::create_directory("output/");
    boost::filesystem::create_directory(outputPrefix);
    FILE *trajectoryFile;
    FILE *currentFile;
    FILE *MSDFile;
    FILE *MSDFileSingleEpsilon;
    FILE *MSDFileSingleContact;
    std::string filename = outputPrefix+"/trajectory";
    trajectoryFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/current";
    currentFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/MSD";
    MSDFile = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/MSDsingleEpsilon";
    MSDFileSingleEpsilon = fopen(filename.c_str(), "w");
    filename = outputPrefix+"/MSDsingleContact";
    MSDFileSingleContact = fopen(filename.c_str(), "w");

    //run simulation for the total simulation time
    while(timePassed < totalSimulationTime){
        doSingleTimeStep();
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

        //output MSD at each time interval tfMSD
        tMSD += dt;
        if (tMSD >= tfMSD) {
            tMSD  = 0.0;
            tfMSD *= 1.1;
            
            //MSD for all partilces
            MSD = 0;
            for(int i = 0; i < N; i++){
                MSD += (x_noPB(i) - x_initial(i)) * (x_noPB(i) - x_initial(i));
            }
            fprintf(MSDFile, "%f, ", timePassed);
            fprintf(MSDFile, "%f\n", MSD/N);

            //MSD for single partilces (distance larger than epsilon)
            MSD = 0;
            for(int i = 0; i < numTaggedEpsilon; i++){
                k = int(round(taggedSingleParticlesEpsilon(i)));
                MSD += (x_noPB(k) - x_initial(k)) * (x_noPB(k) - x_initial(k));
            }
            fprintf(MSDFileSingleEpsilon, "%f, ", timePassed);
            fprintf(MSDFileSingleEpsilon, "%f\n", MSD/numTaggedEpsilon);

            //MSD for single partilces (not in contact)
            MSD = 0;
            for(int i = 0; i < numTaggedContact; i++){
                k = int(round(taggedSingleParticlesContact(i)));
                MSD += (x_noPB(k) - x_initial(k)) * (x_noPB(k) - x_initial(k));
            }
            fprintf(MSDFileSingleContact, "%f, ", timePassed);
            fprintf(MSDFileSingleContact, "%f\n", MSD/numTaggedContact);
        }
    }

    //close output files
    fclose(trajectoryFile);
    fclose(currentFile);
    fclose(MSDFile);
    fclose(MSDFileSingleEpsilon);
    fclose(MSDFileSingleContact);

    //normalize densities
    density /= totalSimulationTime/dt*L*delta;
    density_II /= totalSimulationTime/dt*L*delta*deltaPrime;
    timeOfMerges /= numOfMerges;
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

    for(int i = 0; i < spatialSteps; i++){
        output(i, 1) = density_II(i);
    }
    output.save(outputPrefix+"/density_II", arma::raw_ascii);

    output = arma::zeros<arma::mat>(N, 2);
    for(int i = 0; i < N; i++){
        output(i, 0) = i;
        output(i, 1) = timeOfMerges(i);
    }
    output.save(outputPrefix+"/timeOfMerges", arma::raw_ascii);

    arma::vec meanCS = arma::zeros<arma::vec>(1);
    meanCS(0) = meanClusterSize/totalSimulationTime;
    meanCS.save(outputPrefix+"/meanClusterSize", arma::raw_ascii);
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
 * pure hard spheres (gamma = 0)
 * 
 * @param argc 
 * @param input path to configuration file
 * @return int 
 */
int main(int argc, char** input){
    if(argc > 1){
        cfgFile = input[1];
    }

    loadInput();

    //initialize the simulation
    initialConfiguration();
    //run simulation until equilibrium is reached
    getEquilibriumState();

    //save start time
    auto start = std::chrono::system_clock::now();

    simulateSystem();

    //save end time
    auto end = std::chrono::system_clock::now();
    //calculate the duration of the simulation
    std::chrono::duration<double> elapsed_seconds = end-start;
    //print duration
    std::cout<< "duration of the simulation: " <<elapsed_seconds.count()<<" s"<<std::endl;

    saveResults();
}
