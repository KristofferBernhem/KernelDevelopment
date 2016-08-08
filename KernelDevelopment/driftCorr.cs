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

            int N = 200; // number of gaussians to fit.
            int[] firstDataSet = generateTest(N,0);
            int[] secondDataSet = generateTest(N,15);

                    Stopwatch watch = new Stopwatch();
            watch.Start();
            
            int nrXsteps = 21;
            int nrYsteps = 21;
            int nrZsteps = 21;
            double xyStep = 25;
            double zStep = 25;
            int[] device_firstDataSet       = gpu.CopyToDevice(firstDataSet);
            int[] device_secondDataSet      = gpu.CopyToDevice(secondDataSet);
            double[] device_result      = gpu.Allocate<double>(nrXsteps* nrYsteps* nrZsteps);
            double minDist              = 50*50;
            double[] center             = { 0, 0, 0 };
            double[] device_center      = gpu.CopyToDevice(center);
            // launch kernel. gridsize = N, blocksize = 1. Call once for coarse. Retain datasets on device, update steps and run again centered around minimum.

            gpu.Launch(nrXsteps * nrYsteps * nrZsteps, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_center, xyStep, zStep, minDist, device_result);
            // Collect results.

            double[] result = new double[nrXsteps * nrYsteps * nrZsteps];
            gpu.CopyFromDevice(device_result, result);
            double maxDist = 0;
            int indexing = 0;
            for (int j = 0; j < nrXsteps * nrYsteps * nrZsteps; j++)
            {
                if (maxDist < result[j])
                {
                    indexing = j;
          //          Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
                    maxDist = result[j];
   
                }
            }

            Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
            int dimXY = 21;
            int dimZ = 21;
            double[] drift = new double[3];
            drift[0] = indexing % dimXY;
            drift[1] = indexing / dimXY;
            drift[2] = drift[1] % dimZ;
            drift[1] = drift[1] % dimZ;

            center[0] += -xyStep * ((dimXY - 1) / 2 - drift[0]);
            center[1] += -xyStep * ((dimXY - 1) / 2 - drift[1]);
            center[2] += -zStep * ((dimZ - 1) / 2 - drift[2]);

            Console.WriteLine(center[0]);
            Console.WriteLine(center[1]);
            Console.WriteLine(center[2]);
            
            double[] device_center2 = gpu.CopyToDevice(center);
              
            xyStep /= 10; // finer stepsize.
            zStep /= 10; // finer stepsize.

            gpu.Launch(nrXsteps * nrYsteps * nrZsteps, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_center2, xyStep, zStep, minDist, device_result);

            //double[] result2 = new double[nrXsteps * nrYsteps * nrZsteps];
            gpu.CopyFromDevice(device_result, result);
            maxDist = 0;
            indexing = 0;
            for (int j = 0; j < nrXsteps * nrYsteps * nrZsteps; j++)
            {
                if (maxDist < result[j])
                {
                    indexing = j;
       //             Console.WriteLine("Idx: " + indexing + " distance: " + result2[indexing]);
                    maxDist = result[j];

                }
            }

            Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
            
            drift [0] = indexing % dimXY;
            drift[1] = indexing / dimXY;
            drift[2]  = drift[1] % dimZ;
            drift[1] = drift[1]  % dimZ;

            center[0] += -2.5 * ((dimXY - 1) / 2 - drift[0]);
            center[1] += -2.5 * ((dimXY - 1) / 2 - drift[1]);
            center[2] += -2.5 * ((dimZ - 1) / 2 - drift[2]);

            Console.WriteLine(center[0]);
            Console.WriteLine(center[1]);
            Console.WriteLine(center[2]);
            
            //Profile:
            watch.Stop();
            //  Console.WriteLine("compute time: " + watch.ElapsedMilliseconds);
            Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds);

    
                // Clear gpu.
                gpu.FreeAll();
                gpu.HostFreeAll();
  /*         indexing = 0;
          


                    Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
                    Console.WriteLine("driftcorr x: " + xStepsFine[indexing]);
                    Console.WriteLine("driftcorr y: " + yStepsFine[indexing]);
                    Console.WriteLine("driftcorr z: " + zStepsFine[indexing]);
                    */
            Console.ReadKey(); // keep console up.
        } // Execute()



        [Cudafy]

        public static void driftCorrect(GThread thread, int[] firstDataSet, int[] secondDataSet,double[] center, double xyStep, double zStep, double minDist, double[] result)
        {

            int idx = thread.blockIdx.x;
            if (idx < 9261)
            {
                int dimXY = 21;
                int dimZ = 21;
                // get x y and z coordinates from single vector:
                
                
                
                int x = idx % dimXY;
                int y = idx / dimXY;
                int z = y % dimZ;
                y = y % dimZ;
                double xdist = 0; // distance between particles in x direction.
                double ydist = 0; // distance between particles in y direction.
                double zdist = 0; // distance between particles in z direction.
                double distance = 0; // distance between particles of the two datasets.
                result[idx] = 0;  // as this kernel is called multiple times, reset result vector.

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
        }// end driftCorrect.

    

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