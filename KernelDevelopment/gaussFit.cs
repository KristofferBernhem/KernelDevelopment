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
    class gaussFit
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
            double[] timers= { 0, 0, 0, 0, 0 , 0};
            int count = 0;
//            int[] Ntests = { 100, 1000, 5000, 10000, 20000 };
            int[] Ntests = {10000};
            for (int i = 0; i < Ntests.Length; i++)
            {
                int N = Ntests[i]; // number of gaussians to fit.
                int[] gaussVector = generateGauss(N);
                double convCriteria = 1E-8;
                int maxIterations = 1000;
                //double[] parameterVector = generateParameters(N);
                int windowWidth = 5;            // window for gauss fitting.

                // low-high for each parameter. Bounds are inclusive.
                double[] bounds = {
                              0.6,  1.4,         // amplitude, should be close to center pixel value. Add +/-20 % of center pixel, not critical for performance.
                              1,  windowWidth-1,        // x coordinate. Center has to be around the center pixel if gaussian distributed.
                              1,  windowWidth-1,        // y coordinate. Center has to be around the center pixel if gaussian distributed.
                              0.8,  2.0,        // sigma x. Based on window size.
                              0.8,  2.0,        // sigma y. Based on window size.
                              -1.57, 1.57,        // Theta. 0.785 = pi/4. Any larger and the same result can be gained by swapping sigma x and y, symetry yields only positive theta relevant.
                              -0.5, 0.5};        // offset, best estimate, not critical for performance.
                
                // steps is the most critical for processing time. Final step is 1/25th of these values. 
                double[] steps = {
                                0.1,             // amplitude, make final step 5% of max signal.
                                0.25,           // x step, final step = 1 nm.
                                0.25,           // y step, final step = 1 nm.
                                0.25,            // sigma x step, final step = 2 nm.
                                0.25,            // sigma y step, final step = 2 nm.
                                0.19625,        // theta step, final step = 0.00785 radians. Start value == 25% of bounds.
                                0.01};            // offset, make final step 1% of signal.
                double[] singleParameter = new double[7];
                singleParameter[0] = 6712; // amplitude.
                singleParameter[1] = 2.5;   // x0.
                singleParameter[2] = 2.5;   // y0.
                singleParameter[3] = 1.0;   // sigma x.
                singleParameter[4] = 1.0;   // sigma y.
                singleParameter[5] = 0.0;   // Theta.
                singleParameter[6] = 0;     // offset. 
                double[] parameterVector = generateParameters(singleParameter, N);
                double[] hostSteps = generateParameters(steps, N);

                // Profiling:
                Stopwatch watch = new Stopwatch();
                watch.Start();

                // Transfer data to device. 
                int[] device_gaussVector        = gpu.CopyToDevice(gaussVector);
                double[] device_parameterVector = gpu.CopyToDevice(parameterVector);
                double[] device_bounds          = gpu.CopyToDevice(bounds);
                //double[] device_steps           = gpu.CopyToDevice(steps); // use for old code.
                double[] device_steps           = gpu.CopyToDevice(hostSteps);


                int N_squared = (int)Math.Ceiling(Math.Sqrt(N)); // launch kernel. gridsize = N_squared x N_squared, blocksize = 1.

                //gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitterAdaptive(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps);

                gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitter(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps, convCriteria, maxIterations);
                //gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitter2(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps, convCriteria, maxIterations); // faster with less overhead.
                // Collect results.
                double[] result = new double[7 * N];                // allocate memory for all parameters.
                gpu.CopyFromDevice(device_parameterVector, result); // pull optimized parameters.

                //Profile:
                watch.Stop();
                timers[count] = watch.ElapsedMilliseconds;

                // Clear gpu.
                gpu.FreeAll();
                gpu.HostFreeAll();

                // profiling.
                count++;
                for (int j = 0; j < 7; j ++)
                    Console.WriteLine("P " + j + ": " + result[j]);
            }
            // profiling.
            for (int i = 0; i < Ntests.Length; i++)
                Console.Out.WriteLine(" long variable compute time for : " + Ntests[i] + " particles: " + timers[i] + " ms");

            Console.ReadKey(); // keep console up.
        } // Execute()

        [Cudafy]
        /*
         * Adaptive solver.
         * Taking the starting point, calculate all neighbouring points in parameter space, step to the best improvement and repeat until no improvement can be found. 
         * Follow by decreasing stepsize in all parameter spaces and repeat. Break if total iterations excedeed threshold or no further improvement can be found.
         * Start by optimizing x, y, sigma x, sigma y and theta. Amplitude and offset should not affect these but only final result. These are optimized after the other 5 parameters.
         */
        public static void gaussFitter(GThread thread, int[] gaussVector, double[] P, ushort windowWidth, double[] bounds, double[] stepSize, double convCriteria, int maxIterations)
        {
            int xIdx = thread.blockIdx.x;
            int yIdx = thread.blockIdx.y;

            int idx = xIdx + thread.gridDim.x * yIdx;

            //int idx = thread.blockIdx.x;        // get index for current thread.            
            if (idx < gaussVector.Length / (windowWidth * windowWidth))  // if current idx points to a location in input.
            {
                ///////////////////////////////////////////////////////////////////
                //////////////////////// Setup fitting:  //////////////////////////
                ///////////////////////////////////////////////////////////////////

                int pIdx = 7 * idx;                         // parameter indexing.
                int gIdx = windowWidth * windowWidth * idx; // gaussVector indexing.
                double mx = 0; // moment in x (first order).
                double my = 0; // moment in y (first order).                
                double InputMean = 0;                       // Mean value of input pixels.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                {
                    InputMean   += gaussVector[gIdx + i];
                    mx          += (i % windowWidth) * gaussVector[gIdx + i];
                    my          += (i / windowWidth) * gaussVector[gIdx + i];
                }
                P[pIdx + 1] = mx / InputMean; // weighted centroid as initial guess of x0.
                P[pIdx + 2] = my / InputMean; // weighted centroid as initial guess of y0.
                InputMean = InputMean / (windowWidth * windowWidth); // Mean value of input pixels.

                double totalSumOfSquares = 0;               // Total sum of squares of the gaussian-InputMean, for calculating Rsquare.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                    totalSumOfSquares += (gaussVector[gIdx + i] - InputMean) * (gaussVector[gIdx + i] - InputMean);

                ///////////////////////////////////////////////////////////////////
                //////////////////// intitate variables. //////////////////////////
                ///////////////////////////////////////////////////////////////////
                Boolean optimize = true;
                int loopcounter = 0;
                int xi = 0;
                int yi = 0;
                double residual = 0;
                double Rsquare = 1;
                double oldRsquare = Rsquare;
                int pId = 0;
                double ThetaA = 0;
                double ThetaB = 0;
                double ThetaC = 0;
                double tempRsquare = 0;                
                int xyIndex = 0;
                double photons = 0;
                double ampLowBound = P[pIdx] * bounds[0];  // amplitude bounds are in fraction of center pixel value.
                double ampHighBound = P[pIdx] * bounds[1];  // amplitude bounds are in fraction of center pixel value.
                double offLowBound = P[pIdx] * bounds[12];  // offset bounds are in fraction of center pixel value.
                double offHighBound = P[pIdx] * bounds[13];  // offset bounds are in fraction of center pixel value.
                stepSize[pIdx] *= P[pIdx];
                stepSize[pIdx + 6] *= P[pIdx];
                double sigmX = bounds[6];
                double sigmY = bounds[8];

                while(sigmX <= bounds[7])
                {
                    ThetaA = 1 / (2 * sigmX * sigmX);
                    while(sigmY <= bounds[9])
                    {
                        ThetaC = 1 / (2 * sigmY * sigmY);
                        tempRsquare = 0;
                        for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                        {
                            xi = (xyIndex % windowWidth);
                            yi = (xyIndex / windowWidth);
                            residual = (P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        ))) - gaussVector[gIdx + xyIndex];
                            tempRsquare += residual * residual;
                        } 
                        tempRsquare = tempRsquare / totalSumOfSquares;
                        if (tempRsquare < Rsquare)
                        {
                            Rsquare = tempRsquare;
                            P[pIdx + 3] = sigmX;
                            P[pIdx + 4] = sigmY;
                        }
                        sigmY += stepSize[pIdx + 4];
                    }
                    sigmX += stepSize[pIdx + 3];
                    sigmY = bounds[8];
                }
                           
                Rsquare = 1;   


                // calulating these at this point saves computation time (theta = 0 at this point).

                ThetaA = 1 / (2 * P[pIdx + 3] * P[pIdx + 3]);
                ThetaB = 0;
                ThetaC = 1 / (2 * P[pIdx + 4] * P[pIdx + 4]);



                while (optimize)
                {
                    if (pId == 0) // amplitude
                    {
                        oldRsquare = Rsquare;
                       // if (optimize)          
                        if (P[pIdx + pId] + stepSize[pIdx + pId] > ampLowBound &&
                            P[pIdx + pId] + stepSize[pIdx + pId] < ampHighBound)
                        {
                            P[pIdx + pId] += stepSize[pIdx + pId]; // take one step.

                            tempRsquare = 0; // reset.
                            for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                            {
                                xi = xyIndex % windowWidth;
                                yi = xyIndex / windowWidth;
                                residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + P[pIdx + 6] - gaussVector[gIdx + xyIndex];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < Rsquare)                // If improved, update variables.
                            {
                                Rsquare = tempRsquare;
                            }
                            else
                            {
                                P[pIdx + pId] -= stepSize[pIdx + pId]; // reset.
                                if (stepSize[pIdx + pId] < 0)
                                    if (loopcounter < 20)
                                        stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                    else
                                        stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -1;         // switch direction.
                            }
                        }
                        else // bounds check 
                        {
                            if (stepSize[pIdx + pId] < 0)
                                if (loopcounter < 20)
                                    stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                            else
                                stepSize[pIdx + pId] *= -1;         // switch direction.
                        }
                    }
                    else if (pId == 6) // offset
                    {
                        //if (optimize)          
                        if (P[pIdx + pId] + stepSize[pIdx + pId] > offLowBound &&
                           P[pIdx + pId] + stepSize[pIdx + pId] < offHighBound)
                        {
                            P[pIdx + pId] += stepSize[pIdx + pId]; // take one step.

                            tempRsquare = 0; // reset.
                            for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                            {
                                xi = xyIndex % windowWidth;
                                yi = xyIndex / windowWidth;
                                residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + P[pIdx + 6] - gaussVector[gIdx + xyIndex];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < Rsquare)                // If improved, update variables.
                            {
                                Rsquare = tempRsquare;
                            }
                            else
                            {
                                P[pIdx + pId] -= stepSize[pIdx + pId]; // reset.
                                if (stepSize[pIdx + pId] < 0)
                                    if (loopcounter < 20)
                                        stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                    else
                                        stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -1;         // switch direction.
                            }
                        }
                        else // bounds check 
                        {
                            if (stepSize[pIdx + pId] < 0)
                                if (loopcounter < 20)
                                    stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                            else
                                stepSize[pIdx + pId] *= -1;         // switch direction.
                        }
                    }
                    else // x,y, sigma x, sigma y or theta.
                    {
                        if(optimize)          
                        if ((P[pIdx + pId] + stepSize[pIdx + pId] > bounds[2*pId]) &&
                            (P[pIdx + pId] + stepSize[pIdx + pId] < bounds[2*pId + 1]))
                        {
                            P[pIdx + pId] += stepSize[pIdx + pId]; // take one step.
                            // update sigma and angle dependency.
                            ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            tempRsquare = 0; // reset.
                            for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                            {
                                xi = xyIndex % windowWidth;
                                yi = xyIndex / windowWidth;
                                residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + P[pIdx + 6] - gaussVector[gIdx + xyIndex];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < Rsquare)                // If improved, update variables.
                            {
                                Rsquare = tempRsquare;
                            }
                            else
                            {
                                P[pIdx + pId] -= stepSize[pIdx + pId]; // reset.
                                ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                        Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                                ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) +
                                        Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                                ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                        Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                                if (stepSize[pIdx + pId] < 0)
                                    if (loopcounter < 20)
                                        stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                    else
                                        stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -1;         // switch direction.
                            }
                        } else // bounds check 
                        {
                            if (stepSize[pIdx + pId] < 0)
                                if (loopcounter < 20)
                                    stepSize[pIdx + pId] *= -0.3;   // Decrease stepsize and switch direction.
                                else
                                    stepSize[pIdx + pId] *= -0.7;   // Decrease stepsize and switch direction.
                            else
                                stepSize[pIdx + pId] *= -1;         // switch direction.
                        }
                    }

                    pId++;
                    loopcounter++;

                    if (pId > 6)
                    {
                        if (loopcounter > 250)
                        {
                            if ((oldRsquare - Rsquare) < convCriteria)
                            {
                                optimize = false;
                            }
                        }
                        pId = 0;
                    }                                        
                    if (loopcounter > maxIterations) // exit.
                        optimize = false;
                }// optimize while loop

                ///////////////////////////////////////////////////////////////////
                ///////////////////////// Final output: ///////////////////////////
                ///////////////////////////////////////////////////////////////////
                ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) +
                                    Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                tempRsquare = 0; // reset.
                for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                {
                    xi = xyIndex % windowWidth;
                    yi = xyIndex / windowWidth;
                    residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                            2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                            ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                            )) + P[pIdx + 6];
                    photons += residual;
                    residual -= gaussVector[gIdx + xyIndex];
                    tempRsquare += residual * residual;
                }
                tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                P[pIdx] = photons;          // set amplitude to photon count.
                P[pIdx + 6] = 1 - tempRsquare;  // set offset to r^2;                     
            } //idx check.

        } // gaussFitterAdaptive.
       
        
        
        
   


        /*
         * Generate input for fitter.         
         */
        public static int[] generateGauss(int N)
        {
            int[] gaussVector = new int[7 * 7 * N];
          /*  int[] single_gauss = {
				388, 398,  619,   419, 366,  347, 313,
				638, 819,  1236, 1272, 603,  536, 340, 
				619, 1376, 2153, 2052, 974,  619, 289,
				641, 1596, 2560, 2808, 1228, 449, 240,
				481, 1131, 1537, 1481, 801,  451, 336,
				294, 468,  716,   564, 582,  345, 291,
				278, 316,  451,   419, 347,  276, 291
		};
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};*/
            int[] single_gauss = {627,803,957,716,202,
				763,2061,2678,1531,1134,
				1387,4792,6712,3875,710,
				1365,3558,4858,2676,630,
				1107,1010,906,1144,986				
		};
            for(int i = 0; i < N; i++)
            {
                for(int j = 0; j < single_gauss.Length; j++)
                {
                    gaussVector[i * single_gauss.Length + j] = single_gauss[j];
                }
            }
            return gaussVector;
        }

        /*
         * Generate input for fitter.         
         */
        public static double[] generateParameters(int N)
        {
            double[] parameters = new double[7 * N];
            double[] singleParameter = new double[7];
            singleParameter[0] = 6712; // amplitude.
            singleParameter[1] = 2.5;   // x0.
            singleParameter[2] = 2.5;   // y0.
            singleParameter[3] = 1.5;   // sigma x.
            singleParameter[4] = 1.5;   // sigma y.
            singleParameter[5] = 0.0;   // Theta.
            singleParameter[6] = 0;     // offset.
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < singleParameter.Length; j++)
                {
                    parameters[i * singleParameter.Length + j] = singleParameter[j];
                }
            }
            return parameters;
        }

        /*
         * Replicate input parameter settings and return one copy per N.
         */ 
        public static double[] generateParameters(double[] P, int N)
        {
            double[] parameters = new double[7 * N];
            
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < P.Length; j++)
                {
                    parameters[i * P.Length + j] = P[j];
                }
            }
            return parameters;
        }
    }
}
