using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace KernelDevelopment
{
    class vectorAdd
    {
        /*
         * GTX1080 card: 10k fits in 680 ms, 15x faster then LM and the CPU adaptive version.
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
            int[] vectorA = new int[10000];
            int[] vectorB = new int[10000];
            for (int i = 0; i < 10000; i++)
            {
                vectorA[i] = 1;
                vectorB[i] = 5;
            }
                

                // Transfer data to device. 
            int[] device_vectorA = gpu.CopyToDevice(vectorA);
            int[] device_vectorB = gpu.CopyToDevice(vectorB);


            int[] device_result = gpu.Allocate<int>(vectorA.Length);
                //gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitterAdaptive(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps);

                //gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitter(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps, convCriteria, maxIterations);
            gpu.Launch(new dim3(100, 100), 1).vectorAddTwo(device_vectorA, device_vectorB, device_result); // faster with less overhead.
                // Collect results.
            int[] result = new int[10000];                // allocate memory for all parameters.
            gpu.CopyFromDevice(device_result, result); // pull optimized parameters.

                
                for (int j = 9990; j < 10000; j ++)
                    Console.WriteLine("P " + j + ": " + result[j]);
            
            Console.ReadKey(); // keep console up.
        } // Execute()

        [Cudafy]
        /*
         * Adaptive solver.
         * Taking the starting point, calculate all neighbouring points in parameter space, step to the best improvement and repeat until no improvement can be found. 
         * Follow by decreasing stepsize in all parameter spaces and repeat. Break if total iterations excedeed threshold or no further improvement can be found.
         * Start by optimizing x, y, sigma x, sigma y and theta. Amplitude and offset should not affect these but only final result. These are optimized after the other 5 parameters.
         */
        public static void vectorAddTwo(GThread thread, int[] vectorA, int[] vectorB, int[] vectorC)
        {
            int xIdx = thread.blockIdx.x;
            int yIdx = thread.blockIdx.y;

            int idx = xIdx + thread.gridDim.x * yIdx;

            //int idx = thread.blockIdx.x;        // get index for current thread.            
            if (idx < vectorC.Length) // check range
            {
                vectorC[idx] = vectorA[idx] +vectorB[idx];

            }
        }
    }
}
