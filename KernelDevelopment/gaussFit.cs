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
         * Brute force version:
         * Currently 9-12x faster compared to parallelized CPU version. more speedup most likely possible.
         * 5k fits in ~6s on gtx970 card. Might be furhter improved by looking at kernel call. 
         * 
         * Adaptive version:
         * Currently ~30-40x faster compared to parallelized CPU version. 10k fits in ~6 seconds on gtx970 card. 
         * Further optimization most likely possible.
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
            double[] timers= { 0, 0, 0, 0, 0 };
            int count = 0;
//            int[] Ntests = { 100, 1000, 5000, 10000, 20000 };
            int[] Ntests = { 100, 1000};
            for (int i = 0; i < Ntests.Length; i++)
            {
                int N = Ntests[i]; // number of gaussians to fit.
                int[] gaussVector = generateGauss(N);
                double[] parameterVector = generateParameters(N);
                
                // low-high for each parameter. Bounds are inclusive.
                double[] bounds = {
                              0.7,  1.5,         // amplitude, should be close to center pixel value. Add +/-20 % of center pixel, not critical for performance.
                              1.5,  2.5,        // x coordinate. Center has to be around the center pixel if gaussian distributed.
                              1.5,  2.5,        // y coordinate. Center has to be around the center pixel if gaussian distributed.
                              0.5,  2.5,        // sigma x. Based on window size.
                              0.5,  2.5,        // sigma y. Based on window size.
                                0, .785,        // Theta. 0.785 = pi/4. Any larger and the same result can be gained by swapping sigma x and y, symetry yields only positive theta relevant.
                              -0.25, 0.25};        // offset, best estimate, not critical for performance.
                
                // steps is the most critical for processing time. Final step is 1/25th of these values. 
                double[] steps = {
                                0.25,             // amplitude, make final step 5% of max signal.
                                0.25,           // x step, final step = 1 nm.
                                0.25,           // y step, final step = 1 nm.
                                0.5,            // sigma x step, final step = 2 nm.
                                0.5,            // sigma y step, final step = 2 nm.
                                0.19625,        // theta step, final step = 0.00785 radians. Start value == 25% of bounds.
                                0.025};            // offset, make final step 1% of signal.

                int windowWidth = 5;            // window for gauss fitting.


                // Profiling:
                Stopwatch watch = new Stopwatch();
                watch.Start();

                // Transfer data to device. 
                int[] device_gaussVector        = gpu.CopyToDevice(gaussVector);
                double[] device_parameterVector = gpu.CopyToDevice(parameterVector);
                double[] device_bounds          = gpu.CopyToDevice(bounds);
                double[] device_steps           = gpu.CopyToDevice(steps);


                int N_squared = (int)Math.Ceiling(Math.Sqrt(N)); // launch kernel. gridsize = N_squared x N_squared, blocksize = 1.

                gpu.Launch(new dim3(N_squared, N_squared), 1).gaussFitterAdaptive(device_gaussVector, device_parameterVector, windowWidth, device_bounds, device_steps);
                
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
                Console.Out.WriteLine("compute time for : " + Ntests[i] + " particles: " + timers[i] + " ms");

            Console.ReadKey(); // keep console up.
        } // Execute()

        [Cudafy]
        /*
         * Adaptive solver.
         * Taking the starting point, calculate all neighbouring points in parameter space, step to the best improvement and repeat until no improvement can be found. 
         * Follow by decreasing stepsize in all parameter spaces and repeat. Break if total iterations excedeed threshold or no further improvement can be found.
         * Start by optimizing x, y, sigma x, sigma y and theta. Amplitude and offset should not affect these but only final result. These are optimized after the other 5 parameters.
         */
        public static void gaussFitterAdaptive(GThread thread, int[] gaussVector, double[] P, ushort windowWidth, double[] bounds, double[] stepSize)
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

                double InputMean = 0;                       // Mean value of input pixels.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                    InputMean += gaussVector[gIdx + i];

                InputMean = InputMean / (windowWidth * windowWidth); // Mean value of input pixels.

                double totalSumOfSquares = 0;               // Total sum of squares of the gaussian-InputMean, for calculating Rsquare.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                    totalSumOfSquares += (gaussVector[gIdx + i] - InputMean) * (gaussVector[gIdx + i] - InputMean);

                ///////////////////////////////////////////////////////////////////
                //////////////////// intitate variables. //////////////////////////
                ///////////////////////////////////////////////////////////////////
                Boolean optimize    = true;
                int loopcounter     = 0;
                int xi              = 0;
                int yi              = 0;
                double residual     = 0;
                double Rsquare      = 1;
                double ThetaA       = 0;
                double ThetaB       = 0;
                double ThetaC       = 0;
                double inputRsquare = 0;
                double tempRsquare  = 0;
                double ampStep      = stepSize[0] * P[pIdx];
                double xStep        = stepSize[1];
                double yStep        = stepSize[2];
                double sigmaxStep   = stepSize[3];
                double sigmayStep   = stepSize[4];
                double thetaStep    = stepSize[5];
                double offsetStep   = stepSize[6] * P[pIdx];
                double sigmax2      = 0;
                double sigmay2      = 0;
                double sigmax       = 0;
                double sigmay       = 0;
                double theta        = 0;
                double x            = 0;
                double y            = 0;
                int xyIndex         = 0;
                double photons      = 0;
                double ampLowBound  = P[pIdx] * bounds[0];  // amplitude bounds are in fraction of center pixel value.
                double ampHighBound = P[pIdx] * bounds[1];  // amplitude bounds are in fraction of center pixel value.
                double offLowBound  = P[pIdx] * bounds[12];  // offset bounds are in fraction of center pixel value.
                double offHighBound = P[pIdx] * bounds[13];  // offset bounds are in fraction of center pixel value.
                int stepRefinement  = 0;
                ///////////////////////////////////////////////////////////////////
                /////// optimze x, y, sigma x, sigma y and theta in parallel. /////
                ///////////////////////////////////////////////////////////////////

                while (optimize) 
                {
                inputRsquare = Rsquare; // before loop.                         
                for (sigmax = P[pIdx + 3] - sigmaxStep; sigmax <= P[pIdx + 3] + sigmaxStep; sigmax += sigmaxStep)                        
                    {
                        sigmax2 = sigmax * sigmax; // calulating this at this point saves computation time.
                        for (sigmay = P[pIdx + 4] - sigmayStep; sigmay <= P[pIdx + 4] + sigmayStep; sigmay += sigmayStep)                            
                            {
                            sigmay2 = sigmay * sigmay; // calulating this at this point saves computation time.
                            if (sigmax != sigmay)
                            {
                                for (theta = P[pIdx + 5] - thetaStep; theta <= P[pIdx + 5] + thetaStep; theta += thetaStep)
                                {
                                    if (theta >= bounds[10] && theta <= bounds[11] && // Check that the current parameters are within the allowed range.
                                        sigmax >= bounds[6] && sigmax <= bounds[7] &&
                                        sigmay >= bounds[8] && sigmay <= bounds[9])
                                    {
                                        // calulating these at this point saves computation time.
                                        ThetaA = Math.Cos(theta) * Math.Cos(theta) / (2 * sigmax2) + Math.Sin(theta) * Math.Sin(theta) / (2 * sigmay2);
                                        ThetaB = -Math.Sin(2 * theta) / (4 * sigmax2) + Math.Sin(2 * theta) / (4 * sigmay2);
                                        ThetaC = Math.Sin(theta) * Math.Sin(theta) / (2 * sigmax2) + Math.Cos(theta) * Math.Cos(theta) / (2 * sigmay2);

                                        for (x = P[pIdx + 1] - xStep; x <= P[pIdx + 1] + xStep; x += xStep)
                                        {
                                            for (y = P[pIdx + 2] - yStep; y <= P[pIdx + 2] + yStep; y += yStep)
                                            {
                                                if (x >= bounds[2] && x <= bounds[3] && // Check that the current parameters are within the allowed range.
                                                    y >= bounds[4] && y <= bounds[5])
                                                {
                                                    // Calculate residual for this set of parameters.
                                                    tempRsquare = 0; // reset.
                                                    for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                                                    {
                                                        xi = xyIndex % windowWidth;
                                                        yi = xyIndex / windowWidth;
                                                        residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - x) * (xi - x) -
                                                                2 * ThetaB * (xi - x) * (yi - y) +
                                                                ThetaC * (yi - y) * (yi - y)
                                                                )) + P[pIdx + 6] - gaussVector[gIdx + xyIndex];

                                                        tempRsquare += residual * residual;
                                                    }
                                                    tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                                                    if (tempRsquare < 0.999 * Rsquare)                // If improved, update variables.
                                                    {
                                                        Rsquare = tempRsquare;
                                                        P[pIdx + 1] = x;
                                                        P[pIdx + 2] = y;
                                                        P[pIdx + 3] = sigmax;
                                                        P[pIdx + 4] = sigmay;
                                                        P[pIdx + 5] = theta;

                                                    } // update parameters
                                                }// bounds check.
                                            } // y loop.
                                        } // x loop.
                                    } // Theta check.
                                } //  theta loop.
                            } else // if sigmax and sigmay are the same, theta = 0 as the gaussian is perfectly circular. 
                            {
                                theta = 0;
                                if (sigmax >= bounds[6] && sigmax <= bounds[7] && // Check that the current parameters are within the allowed range.
                                    sigmay >= bounds[8] && sigmay <= bounds[9])
                                {
                                    // calulating these at this point saves computation time.
                                    ThetaA = Math.Cos(theta) * Math.Cos(theta) / (2 * sigmax2) + Math.Sin(theta) * Math.Sin(theta) / (2 * sigmay2);
                                    ThetaB = -Math.Sin(2 * theta) / (4 * sigmax2) + Math.Sin(2 * theta) / (4 * sigmay2);
                                    ThetaC = Math.Sin(theta) * Math.Sin(theta) / (2 * sigmax2) + Math.Cos(theta) * Math.Cos(theta) / (2 * sigmay2);

                                    for (x = P[pIdx + 1] - xStep; x <= P[pIdx + 1] + xStep; x += xStep)
                                    {
                                        for (y = P[pIdx + 2] - yStep; y <= P[pIdx + 2] + yStep; y += yStep)
                                        {
                                            if (x >= bounds[2] && x <= bounds[3] && // Check that the current parameters are within the allowed range.
                                                y >= bounds[4] && y <= bounds[5])
                                            {
                                                // Calculate residual for this set of parameters.
                                                tempRsquare = 0; // reset.
                                                for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                                                {
                                                    xi = xyIndex % windowWidth;
                                                    yi = xyIndex / windowWidth;
                                                    residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - x) * (xi - x) -
                                                            2 * ThetaB * (xi - x) * (yi - y) +
                                                            ThetaC * (yi - y) * (yi - y)
                                                            )) + P[pIdx + 6] - gaussVector[gIdx + xyIndex];

                                                    tempRsquare += residual * residual;
                                                }
                                                tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                                                if (tempRsquare < 0.999 * Rsquare)                // If improved, update variables.
                                                {
                                                    Rsquare = tempRsquare;
                                                    P[pIdx + 1] = x;
                                                    P[pIdx + 2] = y;
                                                    P[pIdx + 3] = sigmax;
                                                    P[pIdx + 4] = sigmay;
                                                    P[pIdx + 5] = theta;

                                                } // update parameters
                                            }// bounds check.
                                        } // y loop.
                                    } // x loop.
                                } // Theta check.


                            }
                        } // sigma y loop.
                    } // sigmax loop.
                    loopcounter++;
                    if (inputRsquare == Rsquare) // if no improvement was made.
                    {
                        if (stepRefinement < 2) // if stepsize has not been decreased twice already.
                        {
                            xStep           = xStep         / 5;
                            yStep           = yStep         / 5;
                            sigmaxStep      = sigmaxStep    / 5;
                            sigmayStep      = sigmayStep    / 5;
                            thetaStep       = thetaStep     / 5;
                            stepRefinement++;
                        }
                        else
                            optimize = false; // exit.
                    }
                    if (loopcounter > 1000) // exit.
                        optimize = false;
                } // optimize while loop.

            /////////////////////////////////////////////////////////////////////////////
            ////// optimize  amplitude and offset. Only used for photon estimate. ///////
            /////////////////////////////////////////////////////////////////////////////

            // no need to recalculate these for offset and amplitude:
            ThetaA      = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
            ThetaB      = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
            ThetaC      = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
            optimize    = true; // reset.
            loopcounter = 0; // reset.
            stepRefinement = 0; // reset.   
            while (optimize) // optimze amplitude and offset.
            {
                inputRsquare = Rsquare; // before loop.
                for (double amp = P[pIdx] - ampStep; amp <= P[pIdx] + ampStep; amp = amp + ampStep)
                {
                    for (double offset = P[pIdx + 6] - offsetStep; offset <= P[pIdx + 6] + offsetStep; offset = offset + offsetStep)
                    {
                        tempRsquare = 0;
                        if (amp > ampLowBound && amp < ampHighBound &&
                            offset > offLowBound && offset < offHighBound)
                        {
                            for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                            {
                                xi = xyIndex % windowWidth;
                                yi = xyIndex / windowWidth;
                                residual = amp * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + offset - gaussVector[gIdx + xyIndex];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < 0.999 * Rsquare)// If improved, update variables.
                            {
                                Rsquare     = tempRsquare;
                                P[pIdx]     = amp;
                                P[pIdx + 6] = offset;

                            } // update parameters
                        }// Check if within bounds.
                    } // offset loop.
                } // amplitude loop.

                loopcounter++;
                if (inputRsquare == Rsquare) // if no improvement was made.
                {
                    if (stepRefinement < 2) // if stepsize has not been decreased twice already.
                    {
                        ampStep = ampStep / 5;
                        offsetStep = offsetStep / 5;
                        stepRefinement++;
                    }
                    else
                        optimize = false; // exit.
                }
                if (loopcounter > 1000) // exit.
                    optimize = false;
                }// optimize while loop
                
                ///////////////////////////////////////////////////////////////////
                ///////////////////////// Final output: ///////////////////////////
                ///////////////////////////////////////////////////////////////////
                
                for (xyIndex = 0; xyIndex < windowWidth * windowWidth; xyIndex++)
                {
                    xi = xyIndex % windowWidth;
                    yi = xyIndex / windowWidth;
                    residual = P[pIdx] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                            2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                            ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                            )) + P[pIdx + 6];
                    photons += residual;
                    residual -= gaussVector[gIdx + xyIndex];
                    tempRsquare += residual * residual;
                }
                tempRsquare     = (tempRsquare / totalSumOfSquares);
                P[pIdx]         = photons;          // set amplitude to photon count.
                P[pIdx + 6]     = 1 - tempRsquare;  // set offset to r^2;                                 
            } //idx check.
                 
        } // gaussFitterAdaptive.
        

        [Cudafy]
        /*
         * Brute force fitter. Not sensitive to local minima.
         */
        public static void gaussFitterBruteForce(GThread thread, int[] gaussVector, double[] P, int windowWidth, double[] bounds, double[] stepSize)
        {        
            int idx = thread.blockIdx.x;
            //idx = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            if (idx < gaussVector.Length / (windowWidth * windowWidth))
            {
                // Setup fitting:
                int pIdx = 7 * idx;                         // parameter indexing.
                int gIdx = windowWidth * windowWidth * idx; // gaussVector indexing.

                double InputMean = 0;                       // Mean value of input gaussian.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                    InputMean += gaussVector[gIdx + i];

                InputMean = InputMean / (windowWidth * windowWidth);

                double totalSumOfSquares = 0;               // Total sum of squares of the gaussian-InputMean, for calculating Rsquare.
                for (int i = 0; i < windowWidth * windowWidth; i++)
                    totalSumOfSquares += (gaussVector[gIdx + i] - InputMean) * (gaussVector[gIdx + i] - InputMean);
                
                // Gauss evaluation:
                int xi          = 0;
                int yi          = 0;
                double residual = 0;
                double Rsquare = 0;
                double ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                double ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                double ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                for (int i = 0; i < windowWidth * windowWidth; i++)
                {
                    xi = i % windowWidth;
                    yi = i / windowWidth;
                    residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                            2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                            ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                            )) + P[pIdx + 6] - gaussVector[gIdx + i];

                    Rsquare += residual * residual;
                }
                Rsquare = (Rsquare / totalSumOfSquares);  // normalize.

                double tempRsquare = 0;
                
                // Stepsizes:
                double amplitudeStep    = (bounds[1] - bounds[0]) / stepSize[0];
                double xStep            = (bounds[3] - bounds[2]) / stepSize[1];
                double yStep            = (bounds[5] - bounds[4]) / stepSize[2];
                double sigmaXStepSize   = (bounds[7] - bounds[6]) / stepSize[3];
                double sigmaYStepSize   = (bounds[9] - bounds[8]) / stepSize[4];
                double thetaStep        = (bounds[11] - bounds[10]) / stepSize[5];
                double offsetStep       = (bounds[13] - bounds[12]) / stepSize[6];
                

                // Optimize fit:
                for (int step = 1; step < 11; step += 9) // decrease stepsize factor 10 second round through.
                {
                    // Update limits:
                    amplitudeStep   = amplitudeStep     / step;
                    xStep           = xStep             / step;
                    yStep           = yStep             / step;
                    sigmaXStepSize  = sigmaXStepSize    / step;
                    sigmaYStepSize  = sigmaYStepSize    / step;
                    thetaStep       = thetaStep         / step;
                    offsetStep      = offsetStep        / step;

                    // Calculate optimimal sigma x and y.
                    for (double sigmax = P[pIdx + 3] - sigmaXStepSize * stepSize[3]; sigmax < P[pIdx + 3] + sigmaXStepSize * stepSize[3]; sigmax = sigmax + sigmaXStepSize)
                    {
                        for (double sigmay = P[pIdx + 4] - sigmaYStepSize * stepSize[4]; sigmay < P[pIdx + 4] + sigmaYStepSize * stepSize[4]; sigmay = sigmay + sigmaXStepSize)
                        {
                            tempRsquare = 0;
                            ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * sigmax * sigmax) + Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * sigmay * sigmay);
                            ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * sigmax * sigmax) + Math.Sin(2 * P[pIdx + 5]) / (4 * sigmay * sigmay);
                            ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * sigmax * sigmax) + Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * sigmay * sigmay);
                            for (int i = 0; i < windowWidth * windowWidth; i++)
                            {
                                xi = i % windowWidth;
                                yi = i / windowWidth;
                                residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + P[pIdx + 6] - gaussVector[gIdx + i];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);    // normalize.
                            if (tempRsquare < Rsquare)                          // Attempt to minimize
                            {
                                P[pIdx + 3] = sigmax; // add currently best value to parameter list.
                                P[pIdx + 4] = sigmay; // add currently best value to parameter list.
                                Rsquare = tempRsquare; // update lowest Rsquare thus far.
                            } // end rSquare evaluation.
                        } // end sigmay evaluation.
                    } // end sigmax evaluation.


                    // Calculate optimal x and y position.
                    for (double x = P[pIdx + 1] - xStep * stepSize[1]; x < P[pIdx + 1] + xStep * stepSize[1]; x = x + xStep)
                    {
                        for (double y = P[pIdx + 2] - yStep * stepSize[2]; y < P[pIdx + 2] + yStep * stepSize[2]; y = y + yStep)
                        {
                            tempRsquare = 0;
                            ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            for (int i = 0; i < windowWidth * windowWidth; i++)
                            {
                                xi = i % windowWidth;
                                yi = i / windowWidth;
                                residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - x) * (xi - x) -
                                        2 * ThetaB * (xi - x) * (yi - y) +
                                        ThetaC * (yi - y) * (yi - y)
                                        )) + P[pIdx + 6] - gaussVector[gIdx + i];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < Rsquare)
                            {
                                P[pIdx + 1] = x;
                                P[pIdx + 2] = y;
                                Rsquare = tempRsquare;
                            } // Rsquare evaluation
                        } // end y evaluation
                    }// end x evaluation


                    // Calculate optimal angle.
                    for (double theta = P[pIdx + 5] - thetaStep * stepSize[5]; theta < P[pIdx + 5] + thetaStep * stepSize[5]; theta = theta + thetaStep)
                    {
                        tempRsquare = 0;
                        ThetaA = Math.Cos(theta) * Math.Cos(theta) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(theta) * Math.Sin(theta) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                        ThetaB = -Math.Sin(2 * theta) / (4 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(2 * theta) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                        ThetaC = Math.Sin(theta) * Math.Sin(theta) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Cos(theta) * Math.Cos(theta) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                        for (int i = 0; i < windowWidth * windowWidth; i++)
                        {
                            xi = i % windowWidth;
                            yi = i / windowWidth;
                            residual = P[pIdx + 0] * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                    2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                    ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                    )) + P[pIdx + 6] - gaussVector[gIdx + i];

                            tempRsquare += residual * residual;
                        }
                        tempRsquare = (tempRsquare / totalSumOfSquares);    // normalize.
                        if (tempRsquare < Rsquare)
                        {
                            P[pIdx + 5] = theta;
                            Rsquare = tempRsquare;
                        } // Rsquare evaluation
                    } // end theta evaluation
                    
                    // Calculate optimal offset and amplitude.
                    for (double offset = P[pIdx + 6] - offsetStep * stepSize[6]; offset < P[pIdx + 6] + offsetStep * stepSize[6]; offset = offset + offsetStep)
                    {
                        for (double amplitude = P[pIdx] - amplitudeStep * stepSize[0]; amplitude < P[pIdx] + amplitudeStep * stepSize[0]; amplitude = amplitude + amplitudeStep)
                        {
                            tempRsquare = 0;
                            ThetaA = Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaB = -Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 3] * P[pIdx + 3]) + Math.Sin(2 * P[pIdx + 5]) / (4 * P[pIdx + 4] * P[pIdx + 4]);
                            ThetaC = Math.Sin(P[pIdx + 5]) * Math.Sin(P[pIdx + 5]) / (2 * P[pIdx + 3] * P[pIdx + 3]) + Math.Cos(P[pIdx + 5]) * Math.Cos(P[pIdx + 5]) / (2 * P[pIdx + 4] * P[pIdx + 4]);
                            for (int i = 0; i < windowWidth * windowWidth; i++)
                            {
                                xi = i % windowWidth;
                                yi = i / windowWidth;
                                residual = amplitude * Math.Exp(-(ThetaA * (xi - P[pIdx + 1]) * (xi - P[pIdx + 1]) -
                                        2 * ThetaB * (xi - P[pIdx + 1]) * (yi - P[pIdx + 2]) +
                                        ThetaC * (yi - P[pIdx + 2]) * (yi - P[pIdx + 2])
                                        )) + offset - gaussVector[gIdx + i];

                                tempRsquare += residual * residual;
                            }
                            tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.
                            if (tempRsquare < Rsquare)
                            {
                                P[pIdx] = amplitude;
                                P[pIdx + 6] = offset;
                                Rsquare = tempRsquare;
                            } // Rsquare evaluation
                        } // end amplitude evaluation
                    } // end offset evaluation
                } // end step loop
            } // idx check end
        }// kernel end

        /*
         * Generate input for fitter.         
         */
        public static int[] generateGauss(int N)
        {
            int[] gaussVector = new int[5 * 5 * N];
            int[] single_gauss = { 
                0  ,12 ,25 ,12 ,0  ,
                12 ,89 ,153,89 ,12 ,
                25 ,153,255,153,25 ,
                12 ,89 ,153,89 ,12 ,
                0  ,12 ,25 ,12 ,0  ,
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
            singleParameter[0] = 255;   // amplitude.
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

    }
}
