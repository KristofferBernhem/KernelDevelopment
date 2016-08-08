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
            int[] secondDataSet = generateTest(N,-243);

            int[] xSteps = new int[21 * 21 * 21];
            int[] ySteps = new int[21 * 21 * 21];
            int[] zSteps = new int[21 * 21 * 21];
            int count = 0;
            for (int x = -250; x <= 250; x = x + 25)
            {
                for (int y = -250; y <= 250; y = y + 25)
                {
                    for (int z = -250; z <= 250; z = z + 25)
                    {
                        xSteps[count] = x;
                        ySteps[count] = y;
                        zSteps[count] = z;
                        count++;
                    }
                }
            }

            Stopwatch watch = new Stopwatch();
            watch.Start();

            int[] device_xSteps = gpu.CopyToDevice(xSteps);
            int[] device_ySteps = gpu.CopyToDevice(ySteps);
            int[] device_zSteps = gpu.CopyToDevice(zSteps);
            int[] device_firstDataSet = gpu.CopyToDevice(firstDataSet);
            int[] device_secondDataSet = gpu.CopyToDevice(secondDataSet);
            double[] device_result = gpu.Allocate<double>(xSteps.Length);
            double minDist = 50*50;
            // launch kernel. gridsize = N, blocksize = 1. Call once for coarse. Retain datasets on device, update steps and run again centered around minimum.
            gpu.Launch(xSteps.Length, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_xSteps, device_ySteps, device_zSteps, minDist, device_result);
                // Collect results.

            double[] result = new double[xSteps.Length];
            gpu.CopyFromDevice(device_result, result);
            double maxDist = 0;
            int xCenter = 0;
            int yCenter = 0;
            int zCenter = 0;
            int indexing = 0;
            for (int j = 0; j < result.Length; j++)
            {
                if (maxDist < result[j])
                {
                    indexing = j;
                    maxDist = result[j];
                    xCenter = xSteps[j];
                    yCenter = ySteps[j];
                    zCenter = zSteps[j];
                }
            }

            Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
            Console.WriteLine("driftcorr x: " + xSteps[indexing]);
            Console.WriteLine("driftcorr y: " + ySteps[indexing]);
            Console.WriteLine("driftcorr z: " + zSteps[indexing]);

            // generate new steps:
            count = 0;
            int[] xStepsFine = new int[26 * 26 * 26];
            int[] yStepsFine = new int[26 * 26 * 26];
            int[] zStepsFine = new int[26 * 26 * 26];
            for (int x = xCenter - 25; x <= xCenter + 25; x = x + 2)
            {
                for (int y = yCenter - 25; y <= yCenter + 25; y = y + 2)
                {
                    for (int z = zCenter - 25; z <= zCenter + 25; z = z + 2)
                    {
                        xStepsFine[count] = x;
                        yStepsFine[count] = y;
                        zStepsFine[count] = z;
                        count++;
                    }
                }
            }
            // remove old:
            gpu.Free(device_xSteps);
            gpu.Free(device_ySteps);
            gpu.Free(device_zSteps);
            // update:
            int[] device_xStepsFine = gpu.CopyToDevice(xStepsFine);
            int[] device_yStepsFine = gpu.CopyToDevice(yStepsFine);
            int[] device_zStepsFine = gpu.CopyToDevice(zStepsFine);
    //        gpu.Launch(xSteps.Length, 1).driftCorrect(device_firstDataSet, device_secondDataSet, device_xStepsFine, device_yStepsFine, device_zStepsFine, minDist, device_result);

            gpu.CopyFromDevice(device_result, result);
            //Profile:
            watch.Stop();
            //  Console.WriteLine("compute time: " + watch.ElapsedMilliseconds);
            Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds);

    
                // Clear gpu.
                gpu.FreeAll();
                gpu.HostFreeAll();
           indexing = 0;
            for (int j = 0; j < result.Length; j++)
            {
                if (maxDist < result[j])
                {
                    indexing = j;
                    maxDist = result[j];
                    xCenter = xStepsFine[j];
                    yCenter = yStepsFine[j];
                    zCenter = zStepsFine[j];
                }
            }

                    Console.WriteLine("Idx: " + indexing + " distance: " + result[indexing]);
                    Console.WriteLine("driftcorr x: " + xStepsFine[indexing]);
                    Console.WriteLine("driftcorr y: " + yStepsFine[indexing]);
                    Console.WriteLine("driftcorr z: " + zStepsFine[indexing]);
                    
            Console.ReadKey(); // keep console up.
        } // Execute()

        [Cudafy]

        public static void driftCorrect(GThread thread, int[] firstDataSet, int[] secondDataSet, int[] xStep, int[] yStep, int[] zStep, double minDist, double[] result)
        {
            
            int idx = thread.blockIdx.x;
            
            if (idx < xStep.Length)
            {
                int xdist = 0; // distance between particles in x direction.
                int ydist = 0; // distance between particles in y direction.
                int zdist = 0; // distance between particles in z direction.
                double distance = 0; // distance between particles of the two datasets.
                result[idx] = 0;  // as this kernel is called multiple times, reset result vector.
                for(int Particle = 0; Particle < firstDataSet.Length; Particle = Particle + 3)// loop over all particles in the first dataset.
                {
                    for (int targetParticle = 0; targetParticle < secondDataSet.Length; targetParticle = targetParticle + 3) // loop over all particles in the second dataset.
                    {
                        xdist = (firstDataSet[Particle] - secondDataSet[targetParticle] - xStep[idx]) * (firstDataSet[Particle] - secondDataSet[targetParticle] - xStep[idx]);
                        if (xdist < minDist)
                        {
                            ydist = (firstDataSet[Particle+1] - secondDataSet[targetParticle+1] - yStep[idx]) * (firstDataSet[Particle+1] - secondDataSet[targetParticle+1] - yStep[idx]);
                            if (ydist < minDist)
                            {
                                zdist = (firstDataSet[Particle + 2] - secondDataSet[targetParticle + 2] - zStep[idx]) * (firstDataSet[Particle + 2] - secondDataSet[targetParticle + 2] - zStep[idx]);
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