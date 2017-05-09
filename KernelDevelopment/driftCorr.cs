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
         //   int[] dimensions = { 600, 600, 1 };
            int[] dimensions = { 8, 8, 1 };
           /* int[] firstFrame = generateTest(drift, dimensions);
            drift[0] = -1;
            drift[1] = 1;
            drift[2] = 0;
            int[] secondFrame = generateTest(drift, dimensions);*/
            int maxShift    = 8;
            int maxShiftZ   = 5;
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
                            shift[counter]     = xShift;
                            shift[counter + 1] = yShift;
                            shift[counter + 2] = zShift;
                            counter += 3;
                        }
                    }else
                    {
                        shift[counter]     = xShift;
                        shift[counter + 1] = yShift;
                        shift[counter + 2] = 0;
                        counter += 3;
                    }
                }
            }
                   

/*            double referenceSquare = 0;
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
            }*/

            int[] data = {      0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 10, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 100, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0};
            
            int[] data2 = {     0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 1, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 100, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 0};
            double referenceSquare =0;
            double[] targetSquare = {0};
            float[] means = { 110 / 64, 101 / 64 };
            for (int i = 0; i < data.Length; i++ )
            {
                if (data[i]>0)
                {
                    referenceSquare += (data[i] - means[0]) * (data[i] - means[0]);
                }
                if (data2[i]>0)
                {
                    targetSquare[0] += (data2[i] - means[1]) * (data2[i] - means[1]);
                }
            }
            referenceSquare = Math.Sqrt(referenceSquare);
            targetSquare[0] = Math.Sqrt(targetSquare[0]);

              dimensions[0] = 8;
            dimensions[1] = 8;
            referenceSquare = Math.Sqrt(referenceSquare);
            Stopwatch watch = new Stopwatch();
            watch.Start();
            // USER INPUT:
            
            /*int[] device_firstFrame = gpu.CopyToDevice(firstFrame);   // Stationary dataset.
            int[] device_secondFrame = gpu.CopyToDevice(secondFrame);  // Target dataset, move this to maximize correlation. */
            int[] device_firstFrame = gpu.CopyToDevice(data);   // Stationary dataset.
            int[] device_secondFrame = gpu.CopyToDevice(data2);  // Target dataset, move this to maximize correlation.
         //   int frameBatch = dimensions[1];
            
            float[] device_means = gpu.CopyToDevice(means);
            int[] device_dimensions = gpu.CopyToDevice(dimensions);   // Stationary dataset.
            
            double[] device_result = gpu.Allocate<double>(shift.Length / 3); // output.                       
            //double[] device_result = gpu.Allocate<double>(dimensions[0] * dimensions[1] * dimensions[2]); // output.                       
            int[] device_shift = gpu.CopyToDevice(shift);   // Stationary dataset.            
            
            int gridSize = (int)Math.Ceiling(Math.Sqrt(dimensions[0] * dimensions[1] * dimensions[2]));
            //int gridSize = (int)Math.Ceiling(Math.Sqrt(shift.Length / 3));     
            gpu.Launch(new dim3(gridSize, gridSize), 1).runAdd(device_firstFrame, device_secondFrame, device_shift, device_means, device_dimensions, device_result);

            gpu.Synchronize();
            double[] result = new double[shift.Length/3];   // initialize.
            gpu.CopyFromDevice(device_result, result);                      // Get results.
            for (int m = 0; m < result.Length; m ++ )
            {
                if (result[m] > 0.1)
                    Console.WriteLine("result " + m + ": " + result[m] /
                       (Math.Sqrt(100)*Math.Sqrt(100)) + " from shift: " + shift[ 3*m] + " x " + shift[1+ 3* m ] + " x " + shift[2  +3*m]);
            }
            watch.Stop();
              Console.WriteLine("Computation time: " + watch.ElapsedMilliseconds + " mean " + means[0]);
              Console.WriteLine(gridSize);
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
        public static void runAdd(GThread thread, int[] referenceDataSet, int[] targetDataSet, int[] shift, float[] means, int[] dimensions, double[] result)
        {
            int idx = thread.blockIdx.x + thread.gridDim.x * thread.blockIdx.y; // get pixel index.
          //  idx *= dimensions[1];
            if (idx < dimensions[0] *dimensions[1]*  dimensions[2]) // if within range.
            {
           /*     int nextIdx = idx + dimensions[1];
                while (idx < nextIdx)
                {
             */       // verify shift ok.
             //       if (referenceDataSet[idx] > 0) // no need to compute if the result is 0.
              //      {
                        int zi = (int)(idx / (dimensions[0] * dimensions[1]));
                        int xi = (int)(idx - zi * dimensions[0] * dimensions[1]);
                        int yi = (int)(xi / dimensions[0]);
                        xi -= (int)(yi * dimensions[0]);
                        for (int shiftIdx = 0; shiftIdx < shift.Length / 3; shiftIdx++)
                        {
                            if (xi - shift[shiftIdx * 3 + 0] >= 0 && xi - shift[shiftIdx * 3 + 0] < dimensions[0] &&
                                yi - shift[shiftIdx * 3 + 1] >= 0 && yi - shift[shiftIdx * 3 + 1] < dimensions[1] &&
                                zi - shift[shiftIdx * 3 + 2] >= 0 && zi - shift[shiftIdx * 3 + 2] < dimensions[2])
                            {
                                int tarIdx = (xi - shift[shiftIdx * 3]) + (yi - shift[shiftIdx * 3 + 1]) * dimensions[0] + (zi - shift[shiftIdx * 3 + 2]) * dimensions[1] * dimensions[0];
                                result[shiftIdx] += (referenceDataSet[idx] - means[0]) * (targetDataSet[tarIdx] - means[1]);
                            }
                        }
                //    }
               /*     idx++;
                }*/
            }    
        }
        
        //[Cudafy]
        /*
         * Call the kernel once per bin comparison, get a return containing correlation for all sets, search through to decide which one was optimal.
         * organize datasets at [x1][y1][z1][x2]...
         * numSteps = [x/y steps][z steps]. (set z step to 1 for 2D data). This HAS to match iterations in (int i = - maxShift[0]; i <= maxShift[0]; i += stepSize[0]).
         
        public static void runAddFrame(GThread thread, int[] referenceDataSet, int[] targetDataSet, int[] shift, float[] means, int[] dimensions, double[] result)
        {
            int idx = thread.blockIdx.x + thread.gridDim.x * thread.blockIdx.y; // get pixel index.

            if (idx < shift.Length/3) // if within range.
            {
                int frameIdx = 0;
                while (frameIdx < referenceDataSet.Length)
                {
                    // verify shift ok.
                    if (referenceDataSet[frameIdx] > 0) // no need to compute if the result is 0.
                    {
                        short zi = (short)(frameIdx / (dimensions[0] * dimensions[1]));
                        short xi = (short)(frameIdx - zi * dimensions[0] * dimensions[1]);
                        short yi = (short)(xi / dimensions[0]);
                        xi -= (short)(yi * dimensions[0]);
                        int tarIdx = (xi - shift[idx * 3]) + (yi - shift[idx * 3 + 1]) * dimensions[0] + (zi - shift[idx * 3 + 2]) * dimensions[1] * dimensions[0];
                        if (tarIdx < targetDataSet.Length)
                            result[idx] += (referenceDataSet[frameIdx] - means[0]) * (targetDataSet[tarIdx] - means[1]);

                        
                    }
                frameIdx++;
                }             
            }
        }*/





        public static int[] generateTest(int[] drift, int[] dimensions)
        {
            int[] testdata = new int[dimensions[0] * dimensions[1] * dimensions[2]];
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
            int l = 20;
            int m = 30;
            int n = 0;

            int idx = (l + drift[0]) + (m + drift[1]) * dimensions[0] + (n + drift[2]) * (dimensions[0] * dimensions[1]);
            testdata[idx] = 1;            
            l = 50;
            m = 30;
            n = 0;
            idx = (l + drift[0]) + (m + drift[1]) * dimensions[0] + (n + drift[2]) * (dimensions[0] * dimensions[1]);
            testdata[idx] = 1;
            
            return testdata;
        }

    }

}
