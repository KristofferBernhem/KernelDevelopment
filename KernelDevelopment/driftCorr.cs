using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
namespace KernelDevelopment
{
    class driftCorr
    {
        /*
         * Testing show that the kernel is functional, strong dependence on number of particles in computation times (should be O(N^2)).
         * First pass runs fast, second pass significantly slower. Ask community for issue.
         * Runtime currently ~ 10ms per particle, 5s for 500 particles, ~4s for 400 ets.
         */
        public static void Execute()
        {
            // Initialize.
            CudafyModule km = CudafyTranslator.Cudafy();
            km.GenerateDebug = true;
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            // cudart.dll must be accessible!
            GPGPUProperties prop = null;
            try
            {
                prop = gpu.GetDeviceProperties(true);
            }
            catch (DllNotFoundException)
            {
                prop = gpu.GetDeviceProperties(false);
            }

            // create particles for drift correction:
            int N               = 100; // number of gaussians to fit.
            int[] firstDataSet  = generateTest(N,0);
            int[] secondDataSet = generateTest(N,15);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            // USER INPUT:
            double limit    = 250;  // xyz limit on drift.
            double xyStep   = 5;    // desired final step.
            double zStep    = 5;    // desired final step.

            // Setup initial search:
            xyStep                      *= 10;                              // Coarse initial sweep.
            zStep                       *= 10;                              // Coarse initial sweep.
            double maxDistance          = 50 * 50;                          // maximum distance to be included.
            int nrXsteps                = 1 + 2 * (int)(limit / xyStep);    // total steps to be taken in x.
            int nrYsteps                = 1 + 2 * (int)(limit / xyStep);    // total steps to be taken in y.
            int nrZsteps                = 1 + 2 * (int)(limit / zStep);     // total steps to be taken in z.
            int[] device_firstDataSet   = gpu.CopyToDevice(firstDataSet);   // Stationary dataset.
            int[] device_secondDataSet  = gpu.CopyToDevice(secondDataSet);  // Target dataset, move this to maximize correlation.
            double[] device_result      = gpu.Allocate<double>(nrXsteps* nrYsteps* nrZsteps); // output.
            
            double[] center             = { 0, 0, 0 };                      // Center of search.
            double[] device_center      = gpu.CopyToDevice(center);         // Tranfer to GPU.

            // launch kernel. gridsize = N, blocksize = 1. Call once for coarse. Retain datasets on device, update steps and run again centered around minimum.
            gpu.Launch(nrXsteps * nrYsteps * nrZsteps, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_center, nrXsteps, nrZsteps, xyStep, zStep, maxDistance, device_result);
            
            // Collect results.
            double[] result = new double[nrXsteps * nrYsteps * nrZsteps];   // initialize.
            gpu.CopyFromDevice(device_result, result);                      // Get results.
            double maxDist  = 0;                                            // Maximize this variable.
            int indexing    = 0;                                            // Index of result vector yielding strongest correlation.
            for (int j = 0; j < nrXsteps * nrYsteps * nrZsteps; j++)
            {
                if (maxDist < result[j])
                {
                    indexing    = j;
                    maxDist     = result[j];
   
                }
            }

            // Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]); // display results from first pass.
            double[] drift  = new double[3];                                // optimal drift from first, coarse, pass.
            // translate index to x, y and z drift.
            drift[0]        = indexing % nrXsteps;                          // drift in x.
            drift[1]        = indexing / nrXsteps;                          // drift in y.
            drift[2]        = drift[1] % nrZsteps;                          // drift in z.
            drift[1]        = drift[1] % nrZsteps;                          // drift in y.

            center[0]       = -xyStep * ((nrXsteps - 1) / 2 - drift[0]);   // Calculate new center in x.
            center[1]       = -xyStep * ((nrXsteps - 1) / 2 - drift[1]);   // Calculate new center in y.
            center[2]       = -zStep  * ((nrZsteps - 1) / 2 - drift[2]);   // Calculate new center in z.

    //        Console.WriteLine(center[0]);
   //         Console.WriteLine(center[1]);
   //         Console.WriteLine(center[2]);
            
            double[] device_center2 = gpu.CopyToDevice(center);             // Transfer new center coordinates to gpu.
              
            xyStep                  /= 10; // finer stepsize.
            zStep                   /= 10; // finer stepsize.
            // Launch kernel again with finer stepsize and optimized center.
            gpu.Launch(nrXsteps * nrYsteps * nrZsteps, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_center2, nrXsteps, nrZsteps, xyStep, zStep, maxDistance, device_result);


            gpu.CopyFromDevice(device_result, result);  // get results from gpu.
            maxDist = 0;            // reset.
            indexing = 0;           // reset.
            for (int j = 0; j < nrXsteps * nrYsteps * nrZsteps; j++)
            {
                if (maxDist < result[j])
                {
                    indexing    = j;    
                    maxDist     = result[j];

                }
            }

            //            Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);

            // translate index to x, y and z drift.
            drift[0] = indexing % nrXsteps;                          // drift in x.
            drift[1] = indexing / nrXsteps;                          // drift in y.
            drift[2] = drift[1] % nrZsteps;                          // drift in z.
            drift[1] = drift[1] % nrZsteps;                          // drift in y.

            // Add drift to current center results to finetune drift.
            center[0] += -xyStep * ((nrXsteps - 1) / 2 - drift[0]);   // Calculate new center in x.
            center[1] += -xyStep * ((nrXsteps - 1) / 2 - drift[1]);   // Calculate new center in y.
            center[2] += -zStep * ((nrZsteps - 1) / 2 - drift[2]);   // Calculate new center in z.

//            Console.WriteLine(center[0]);
//            Console.WriteLine(center[1]);
//            Console.WriteLine(center[2]);
            
            //Profile:
            watch.Stop();
            Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds);

            // Clear gpu.
            gpu.FreeAll();
            gpu.HostFreeAll();

            Console.ReadKey(); // keep console up.
        } // Execute()



        [Cudafy]

        public static void driftCorrect(GThread thread, int[] firstDataSet, int[] secondDataSet,double[] center, int dimXY, int dimZ, double xyStep, double zStep, double minDist, double[] result)
        {

            int idx = thread.blockIdx.x;
            if (idx < dimXY* dimXY* dimZ)
            {
                // get x y and z coordinates from single vector:
                int x   = idx % dimXY;
                int y   = idx / dimXY;
                int z   = y % dimZ;
                y       = y % dimZ;

                double xdist    = 0; // distance between particles in x direction.
                double ydist    = 0; // distance between particles in y direction.
                double zdist    = 0; // distance between particles in z direction.
                double distance = 0; // distance between particles of the two datasets.
                result[idx]     = 0; // as this kernel is called multiple times, reset result vector.

                // calculate offset:
                double lambdaX = center[0] - xyStep * ((dimXY - 1) / 2 - x );
                double lambdaY = center[1] - xyStep * ((dimXY - 1) / 2 - y );
                double lambdaZ = center[2] - zStep * ((dimZ - 1) / 2 - z);

                for (int Particle = 0; Particle < firstDataSet.Length; Particle = Particle + 3)// loop over all particles in the first dataset.
                {
                    for (int targetParticle = 0; targetParticle < secondDataSet.Length; targetParticle = targetParticle + 3) // loop over all particles in the second dataset.
                    {
                        xdist = (firstDataSet[Particle] - secondDataSet[targetParticle] - lambdaX) * (firstDataSet[Particle] - secondDataSet[targetParticle] - lambdaX);
                        if (xdist < minDist)
                        {
                            ydist = (firstDataSet[Particle + 1] - secondDataSet[targetParticle + 1] - lambdaY) * (firstDataSet[Particle + 1] - secondDataSet[targetParticle + 1] - lambdaY);
                            if (ydist < minDist)
                            {
                                zdist = (firstDataSet[Particle + 2] - secondDataSet[targetParticle + 2] - lambdaZ) * (firstDataSet[Particle + 2] - secondDataSet[targetParticle + 2] - lambdaZ);
                                if (zdist < minDist)
                                {
                                    distance = xdist + ydist + zdist;
                                    if (distance == 0)
                                        result[idx] += 1;
                                    else
                                        result[idx] += 1 / distance;
                                }
                            }
                        }
                    }
                }                
            }
        }   // end driftCorrect.

    

        public static int[] generateTest(int N, int drift)
        {
            int[] testdata = new int[N * 3];
            int[] xyz = { 50+drift, 50+drift, 50+drift };
            for(int i = 0; i < N; i++)
            {
                for(int j = 0; j < 3; j ++)
                {
                    testdata[i * 3 + j] = xyz[j];
                }
            }

            return testdata;
        }

  
    }

}