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
    class driftCorrOld
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
            int N               = 1000; // number of gaussians to fit.
            int[] firstDataSet  = generateTest(N,0,2);
            int[] secondDataSet = generateTest(N,15,2);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            // USER INPUT:
            int[] numSteps = { 41, 1 };
            int[] stepSize = { 5, 10 };
            int[] maxShift = { 100, 100 };
            int[] device_firstDataSet   = gpu.CopyToDevice(firstDataSet);   // Stationary dataset.
            int[] device_secondDataSet  = gpu.CopyToDevice(secondDataSet);  // Target dataset, move this to maximize correlation.
            int[] device_maxShift = gpu.CopyToDevice(maxShift);   // Stationary dataset.
            int[] device_numSteps = gpu.CopyToDevice(numSteps);   // Stationary dataset.
            int[] device_stepSize = gpu.CopyToDevice(stepSize);   // Stationary dataset.
            int[] device_result = gpu.Allocate<int>(numSteps[0] * numSteps[0] * numSteps[1]); // output.
            
            double[] center             = { 0, 0, 0 };                      // Center of search.
            double[] device_center      = gpu.CopyToDevice(center);         // Tranfer to GPU.

            // launch kernel. gridsize = N, blocksize = 1. Call once for coarse. Retain datasets on device, update steps and run again centered around minimum.
            int gridSize = (int)Math.Ceiling(Math.Sqrt(numSteps[0] * numSteps[0] * numSteps[1]));
            gpu.Launch(new dim3(gridSize, gridSize), 1).run(device_firstDataSet, device_secondDataSet, device_maxShift, device_stepSize, device_numSteps, device_result);
            // Collect results.
            int[] result = new int[numSteps[0] * numSteps[0] * numSteps[1]];   // initialize.
            gpu.CopyFromDevice(device_result, result);                      // Get results.

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] > 0)
                {
                    float x = (maxShift[0] - (i / numSteps[0]) * stepSize[0]);
                    float y = (maxShift[0] - (i % numSteps[0]) * stepSize[0]);
                    Console.WriteLine("line: " + i + ": " + result[i] + " from" + x + " x " + y);                                        
                }
            }
            watch.Stop();
            Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds);
            
            // Clear gpu.
            gpu.FreeAll();
            gpu.HostFreeAll();

            Console.ReadKey(); // keep console up.
        } // Execute()


        [Cudafy]
        /*
         * Call the kernel once per bin comparison, get a return containing correlation for all sets, search through to decide which one was optimal.
         * organize datasets at [x1][y1][z1][x2]...
         * numSteps = [x/y steps][z steps]. (set z step to 1 for 2D data). This HAS to match iterations in (int i = - maxShift[0]; i <= maxShift[0]; i += stepSize[0]).
         */ 
        public static void run(GThread thread, int[] firstDataSet, int[] secondDataSet, int[] maxShift,int[] stepSize, int[] numSteps, int[] result)
        {
            int y = thread.blockIdx.y;
            int x = thread.blockIdx.x;
            int idx = x + (y * thread.gridDim.x);          //  combo.  
            if (idx < numSteps[0] * numSteps[0] * numSteps[1]) // one thread per combo of shifts.
            {
                result[idx] = 0;
                // loop as: idx[0]-idx[numSteps[0]] with x = - maxShift and if 2D y changing. if 3D, x and y fixed whilst z changes.
                if(numSteps[1] == 1) // 2D data
                {
                    float xShift = (maxShift[0] - (idx / numSteps[0]) * stepSize[0]);
                    float yShift = (maxShift[0] - (idx % numSteps[0]) * stepSize[0]);

                    for (int i = 0; i < firstDataSet.Length; i += 2 )
                    {
                        for (int j = 0; j < secondDataSet.Length; j += 2)
                        {
                            if (Math.Abs(firstDataSet[i] - secondDataSet[j] - xShift) < 5 && // x dimension.
                                Math.Abs(firstDataSet[i + 1] - secondDataSet[j + 1] - yShift) < 5) // y dimension.
                                result[idx]++;
                        }
                    }
                }
                else // 3D data.
                {
                    float xShift = (maxShift[0] - (idx/(numSteps[0]*numSteps[1])) * stepSize[0]);
                    float yShift = (maxShift[0] - (idx / numSteps[1]) * stepSize[0]); // change once per full run through of z.
                    float zShift = (maxShift[1] - (idx % numSteps[1])* stepSize[1]); // keep changing once per idx.                    
                    for (int i = 0; i < firstDataSet.Length; i += 3)
                    {
                        for (int j = 0; j < secondDataSet.Length; j += 3)
                        {
                            if (Math.Abs(firstDataSet[i] - secondDataSet[j] - xShift) < 5 && // x dimension.
                                Math.Abs(firstDataSet[i + 1] - secondDataSet[j + 1] - yShift) < 5 && // y dimension.
                                Math.Abs(firstDataSet[i + 2] - secondDataSet[j + 2] - zShift) < 5) // z dimension.                                
                                result[idx]++;
                        }
                    }
                }
            } // idx check
        }   // end driftCorrect.
            

        public static int[] generateTest(int N, int drift, int dimensions)
        {
            int[] testdata = new int[N * dimensions];
            int[] xyz = { 50 + drift, 50 - 2*drift, 50 + 3*drift };
            for(int i = 0; i < N; i++)
            {
                for (int j = 0; j < dimensions; j++)
                {
                    testdata[i * dimensions + j] = xyz[j];
                }
            }

            return testdata;
        }

  
    }

}