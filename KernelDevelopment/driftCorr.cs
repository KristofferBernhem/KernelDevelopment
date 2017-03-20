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

            int[] drift = { 0, 0, 0 };
            //int[] dimensions = { 2*1280, 2*1280, 1 };
            int[] dimensions = { 512, 512, 1 };
            byte[] firstFrame = generateTest(drift, dimensions);
            drift[0] = -1;
            drift[1] = 1;
            drift[2] = 0;
            byte[] secondFrame = generateTest(drift, dimensions);
            int maxShift    = 10;
            int maxShiftZ   = 10;
            int reduce = 1;
            if (dimensions[2] == 1)
            {
                maxShiftZ = 1;
                reduce = 2;
            }
            int[] shift = new int[(24 / reduce) * maxShift * maxShift * maxShiftZ];
            int counter = 0;
            for (int xShift = - maxShift; xShift < maxShift; xShift++)
            {
                for (int yShift = - maxShift; yShift < maxShift; yShift++)
                {
                    if (dimensions[2] > 1)
                    {
                        for (int zShift = -maxShiftZ; zShift < maxShiftZ; zShift++)
                        {
                            shift[counter]      = xShift;
                            shift[counter + 1] = yShift;
                            shift[counter + 2] = zShift;
                            counter += 3;
                        }
                    }else
                    {
                        shift[counter] = xShift;
                        shift[counter + 1] = yShift;
                        shift[counter + 2] = 0;
                        counter += 3;
                    }
                }
            }
                   

            double referenceSquare = 0;
            float mean = 0;
            for (int idx = 0; idx < firstFrame.Length; idx++)
            {
                if(firstFrame[idx]>0)
                    mean++;
            }

            mean = mean / (dimensions[0] * dimensions[1] * dimensions[2]);
            double[] targetSquare = new double[shift.Length / 3];
            for (int idx = 0; idx < firstFrame.Length; idx++)
            {
               referenceSquare += (firstFrame[idx]-mean)*(firstFrame[idx]-mean);

               for (int i = 0; i < shift.Length; i += 3)
               {
                   short zi = (short)(idx / (dimensions[0] * dimensions[1]));
                   int idx2 = idx - zi * dimensions[0] * dimensions[1];
                   short yi = (short)(idx / dimensions[0]);
                   idx2 -= yi * dimensions[0];

                   int tarIdx = (idx - shift[i]) + (yi - shift[i + 1]) * dimensions[0] + (zi - shift[i + 2]) * dimensions[1] * dimensions[0];
                   if (tarIdx >= 0 && tarIdx < secondFrame.Length)
                   {
                       targetSquare[i/3] += (secondFrame[tarIdx] - mean) * (secondFrame[tarIdx] - mean);
                   }
               }
            }
            for (int i = 0; i < targetSquare.Length; i++)
            {
                targetSquare[i] = Math.Sqrt(targetSquare[i]);          
            }
            
            int index = 10 + 4 * dimensions[0] + 6 * dimensions[1] * dimensions[0];
   
            referenceSquare = Math.Sqrt(referenceSquare);
            Stopwatch watch = new Stopwatch();
            watch.Start();
            // USER INPUT:
            float[] means = { mean, mean };
            byte[] device_firstFrame = gpu.CopyToDevice(firstFrame);   // Stationary dataset.
            byte[] device_secondFrame = gpu.CopyToDevice(secondFrame);  // Target dataset, move this to maximize correlation.
            float[] device_means = gpu.CopyToDevice(means);
            int[] device_dimensions = gpu.CopyToDevice(dimensions);   // Stationary dataset.
            int gridSize = (int)Math.Ceiling(Math.Sqrt(dimensions[0]*dimensions[1]*dimensions[2]));
             
            double[] device_result = gpu.Allocate<double>(shift.Length / 3); // output.                       
            int[] device_shift = gpu.CopyToDevice(shift);   // Stationary dataset.
            //gpu.Launch(new dim3(dimensions[0] * (int)Math.Sqrt(dimensions[2]), dimensions[1] * (int)Math.Sqrt(dimensions[2]), 1), 1).runSep(device_firstFrame, device_secondFrame, device_shift, device_means, device_dimensions, device_result);
            //gpu.Launch(new dim3(dimensions[0], dimensions[1], 1), 1).runSep(device_firstFrame, device_secondFrame, device_shift, device_means, device_dimensions, device_result);
            gpu.Launch(new dim3(dimensions[0], dimensions[1], 1), 1).runAdd(device_firstFrame, device_secondFrame, device_shift, device_means, device_dimensions, device_result);
            double[] result = new double[shift.Length / 3];   // initialize.
            gpu.CopyFromDevice(device_result, result);                      // Get results.
            
            for (int m = 0; m < result.Length; m ++ )
            {
                if (result[m] > 0.1)
                    Console.WriteLine("result: " + m + ": " + result[m] / (targetSquare[m] * referenceSquare) + " from shift: " + shift[3 * m] + " x " + shift[1 + 3 * m ] + " x " + shift[2 + 3 * m]);
            }
            watch.Stop();
            Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds);

            Console.WriteLine("total: " + result[0] + "vs " + result[1]);
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
        public static void runAdd(GThread thread, byte[] referenceDataSet, byte[] targetDataSet, int[] shift, float[] means, int[] dimensions, double[] result)
        {
            //int idx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x + thread.blockIdx.y * thread.blockDim.x + thread.threadIdx.y;
            int idx = thread.blockIdx.x * thread.gridDim.x + (thread.blockIdx.y * thread.gridDim.y);          // which pixel.  
            if (idx < dimensions[0] * dimensions[1] * dimensions[2])
            {
                result[0]++;// += idx;
            }
            else
                result[1] += idx;

        }
        [Cudafy]
        /*
         * Call the kernel once per bin comparison, get a return containing correlation for all sets, search through to decide which one was optimal.
         * organize datasets at [x1][y1][z1][x2]...
         * numSteps = [x/y steps][z steps]. (set z step to 1 for 2D data). This HAS to match iterations in (int i = - maxShift[0]; i <= maxShift[0]; i += stepSize[0]).
         */
        public static void runSep(GThread thread, byte[] referenceDataSet, byte[] targetDataSet, int[] shift, float[] means, int[] dimensions, double[] result)
        {
            //int idx = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x + thread.blockIdx.y * thread.blockDim.x + thread.threadIdx.y;
            int idx = thread.blockIdx.x + (thread.blockIdx.y * thread.gridDim.x);          // which pixel.  
            if (idx < dimensions[0] * dimensions[1] * dimensions[2])
            {
                // calculate x-y-z coordiantes for this index.              
                short zi = (short)(idx / (dimensions[0] * dimensions[1]));
                idx -= zi * dimensions[0] * dimensions[1];
                short yi = (short)(idx / dimensions[0]);
                idx -= yi * dimensions[0];
                double refVal = (referenceDataSet[idx + yi * dimensions[0] + zi * dimensions[1] * dimensions[0]] - means[0]);
                int counter = 0;
                for (int i = 0; i < shift.Length - 2; i += 3)
                {
                    int tarIdx = (idx - shift[i]) + (yi - shift[i + 1]) * dimensions[0] + (zi - shift[i + 2]) * dimensions[1] * dimensions[0];
                    if (tarIdx >= 0 && tarIdx < targetDataSet.Length)
                    {
                        double tarDiff = targetDataSet[tarIdx] - means[1];
                        if (tarDiff != 0)
                        {
                            result[counter] += refVal * tarDiff;
                       //     result[counter + 1] += tarDiff * tarDiff;
                        }
                    }
                    counter++;
                }
            }                        
        }

        public static byte[] generateTest(int[] drift, int[] dimensions)
        {
            byte[] testdata = new byte[dimensions[0]*dimensions[1]*dimensions[2]];
            for(int i = 0; i < dimensions[0]; i++)
            {
                for (int j = 0; j < dimensions[1]; j++)
                {
                    for (int k = 0; k < dimensions[2]; k++)
                    {
                        testdata[i + j * dimensions[0] + k * dimensions[1] * dimensions[0]] = 0;
                    }
                }                
            }
            int l = 50;
            int m = 30;
            int n = 0;

            int idx = (l + drift[0]) + (m + drift[1]) * dimensions[0] + (n + drift[2]) * (dimensions[0] * dimensions[1]);
            testdata[idx] = 1;

            l = 30;
            m = 75;
            n = 0;
            idx = (l + drift[0]) + (m + drift[1]) * dimensions[0] + (n + drift[2]) * (dimensions[0] * dimensions[1]);
            testdata[idx] = 1;

            return testdata;
        }

    }

}
