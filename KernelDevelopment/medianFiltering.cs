using System;
using System.Collections;
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
    class medianFiltering
    {
        /*
         * Bug tested with continously increasing values.
         * TODO: test with decreasing values and check possible optimization.
         */ 
        public static void Execute()
        {
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
            // Setup
            int[] answer = { 5766, 655, -229, -372, -294, 432, -280, -187, -593, -970, 81, 335, 366, 628, 277, 1173, -231, -1571, -292, -226, 943, -1300, 1110, -38, -240, -368, -1797, -907, -1042, -1819, -1202, -1567, -619, -1099, -91, -392, 1554, 186, 961, -1585, 986, -323, -1049, 1262, 678, -1078, 947, 204, 1024, 96, 981, 1390, 1728, -203, 375, 183, 1271, -519, -110, -254, -880, -2022, -1090, -2149, -1625, -1302, 837, -1286, -773, -1688, -1719, -1978, -1829, -2328, -282, -849, -26, -1735, -81, -1675, 244, -2755, 215, -1054, -1623, -2163, -271, -2165, -7, -2720, -471, -1908, 1448, -3028, -283, -1910, -265, -979, -392, -1929, -700, -1084, -1448, 737, 165, 4340, 1124, 1612, 167, 175, 3080, 1708, 5727, 4838, 6432, 6016, 5802, 6008, 1787, 3328, 4461, 7494, 4666, 285, -23, 97, 3844, 3898, 1895, 1488, 62, 312, -326, -315, 296, 23, 749, 159, 921, 300, 2987, 4212, 3143, 1184, 1517, -428, -1137, -999, -1042, 188, -64, -292, 37, 2858, 4074, 6866, 3586, 3454, 8015, 1836, -1903, -1237, 3018, -144, -917, 1034, 285, 87, -1225, -1051, -71, -1472, -2, -664, -1090, 682, 15, -715, -764, -397, -209, -396, -2178, 356, 361, -270, 1646, 488, -281, 798, 222, 1045, -594, 1185, 3046, 1488, 1070, 1667, 1285, 541, 1939, 2210, 960, 2245, 1670, 1406, 983, 1866, 1141, 1735, 1571, 1684, 1674, 590, 1674, 49, 0, 15, -515, 426, 3563, 1778, 3574, 62, -79, 540, -500, 944, -399, 450, 890, 687, 471, -695, 570, 1408, 1010, 1624, 110, 1462, 881, 1741, -75, 85, -33, 479, -91, 1671, -729, 1619, -670, -400, -857, -2511, -1134, 172, -1238, -897, -43, 1075, 472, -115, 1635, 19, -956, -307, 302, 664, 1874, 891, 230, 181, 481, 115, 1236, 1869, -563, -1081, 91, 315, 450, 357, -492, 951, 385, 559, 1078, 570, 1503, -296, 284, -742, -1288, -627, -650, 245, 1530, 1651, 3221, 3395, 704, -1452, 392, 4046, 126, -2066, -1066, 1849, 1711, 2773, -4, 303, -513, -895, -472, -538, 0, 270, 914, 696, -135, 962, 410, 629, 548, 373, -145, 1131, 835, 684, -168, -536, 161, 460, 954, 3260, 4113, 155, 164, -477, 403, -504, -856, -604, 1193, 189, -700, -573, 1082, -220, -1147, 128, -1074, 276, 141, -483, 2162, 1329, -393, 1406, -5, 798, -49, 731, -72, 108, 1036, 87, 421, -256, 464, 583, 3282, 2448, 4818, 1841, -700, -90, 45, -541, -385, 779, 1166, 380, 677, 217, -113, 530, 110, 875, 4, -875, -219, 1038, -207, 3884, 3927, 6627, 5044, 1149, -45, -766, -28, 293, 665, 662, -103, -924, 198, 2343, 4169, 4374, 2428, 4639, 3027, 4558, 4160, 5111, 4183, 2569, 4447, 1869, 2312, 1129, 754, 269, -101, -951, -255, -377, -467, 17, 342, -57, 486, 94, 41, -1127, 234, -184, 212, -236, 680, -165, -540, 185, 801, -232, 704, -746, 1950, -378, -171, -772, -763, 1208, 1209, 752, 850, -134, -452, 337, 863, 96, 234, 1008, 1160, 800, 2602, 1804, 770, 436, -439, -264, 912, 389, 470, 427, 460, -173, -525, -111, 955, 182, 481, 1165, 14, 204, -1243, -718, 980, -559, -846, -511, -209, 273, -281, -1408, -52, 351, -330, -688, 327, -166, 763, 53, 843, 1227, -132, 119, 107, 799, 1739, 454, -212, 168, 1198, 1621, -482, -454, 668, 387, 386, 2637, -502, -653, 135, -643, -70, -870, 254, 44, -40, -553, 16, -546, 0, 1549, 292, -136, 1283, -41, 842, -94, 1184, -80, 87, 1911, 3678, 3349, 4014, 1721, 2646, 3006, 370, -49, -531, 524, 1217, 662, 173, 81, 1315, 0, -4, -166, -481, -762, 835, 295, 977, 548, 1025, 511, 115, -495, -935, -56, 38, 0, 634, 1069, 1255, 1379, 937, 766, -620, -323, 901, 1773, -752, 1114, 1092, 1386, 1008, 240, 1385, -71, 517, 396, 350, -1836, 558, 679, 751, 1991, -622, 6, 153, 132, -921, -507, -593, 243, -810, -843, 284, -758, 1174, 629, 583, 1286, 505, 639, -625, 496, 389, 948, 452, 1314, 347, -958, 84, -740, -131, -518, -1459, -479, -1171, 249, -130, -769, -156, 11, 1172, 719, 69, 231, 138, 388, -1157, -1247, -783, 33, -556, -280, -223, -604, 1120, 5238, 3307, 1758, 2912, 4410, 3088, 1072, 56, 394, -481, -133, -160, 687, 871, 293, -328, 430, 331, -1365, -304, 253, 463, 166, 717, 1032, 660, 1152, 1599, 1236, 2216, 904, 1654, 1738, 743, 2378, 3633, 2614, 3093, 996, 699, -533, -35, -299, 93, -414, 1814, 816, 488, -255, -12, -224, 836, 248, -815, 719, 1598, -27, 4, -241, 29, 1015, 69, 66, 175, 302, -1157, 48, 895, 355, -592, -235, 94, -447, 541, 7, 1084, -239, 0, -94, -721, 193, -293, -460, 292, 153, 462, -284, -666, -341, -754, 822, -353, 155, 2882, 1437, 754, 2032, 1242, -413, -663, -467, 465, -811, -1290, 272, 118, -307, -47, -102, 172, 487, 743, 368, 153, 234, 389, 256, 274, 303, -16, 530, 1388, 1304, 150, 233, 1127, 1101, 105, 612, 962, -32, 91, 1179, 237, 326, 192, -36, -1070, -345, 645, 550, 1224, 504, 505, 744, 1146, 873, -141, 437, -438, 103, 52, -305, 658, 428, 486, 341, -210, 345, -556, -345, -401, -523, -362, -994, -228, 565, 55, -118, 3533, 4419, 3920, 1410, 366, 487, 863, 364, 459, 708, 1289, 1099, 1326, 225, -45, 1201, 704, 941, 328, 530, 187, 82, 374, 18, -245, 61, 3410, 593, -955, 166, 233, -1, -815, 512, -545, -375, -214, -49, -353, -975, -358, -901, 338, 1277, 765, 191, 1966, 1653, -237, 560, -354, 543, 506, -126, -449, -281, 928, 309, 429, -232, 450, 604, 1592, 1504, 718, 76, 999, 1800, 1821, 1338, 964, -565, 992, 794, 56, 470, 635, 105, 12, -788, -500, -851, 557, -28, -766, 644, 463, 139, 556, 815, 15, 634, 15, -396, 1296, 855, 640, 1126, 158, 586, -740, 592, -86, 10, 1161, 252, -145, 749, -63, 79, -290, 1177, 410, 942, 2088, 1822, 2259, 3684, 870, 1144, 126, 1109, 1670, 574, 1859, 913, 582, 1634, 3269, 768, 510, 375, -390, 153, -36, 40, 10, -147, -762, -682, -228, 912, -802, -679, -869, -427, -625, -1327, 348, -668, 286, 308, -752, 156, -876, 1536, 104, -119, 845, 113, -488, 236, -301, -426, 1302, 646, -23, 658, 394, 2357, -3, 676, -293 };
            int width           = 50;                                       // filter window width.
            int depth = answer.Length;                                    // z.
            int framewidth      = 64;
            int frameheight     = 64;
            int N               = depth * framewidth * frameheight;         // size of entry.
            float[] meanVector    = medianFiltering.generateTest(depth);      // frame mean value.
            float[] test_data     = medianFiltering.generateTest(N, depth);   // pixel values organized as: x1y1z1, x1y1z2,...,x1y1zn, x2y1z1,...                                                               
            // Profiling:
            Stopwatch watch     = new Stopwatch();
            watch.Start();

            // Transfer data to device.
            float[] device_data       = gpu.CopyToDevice(test_data);
            float[] device_meanVector = gpu.CopyToDevice(meanVector);
            int[] device_result     = gpu.Allocate<int>(N);
            float[] device_window     = gpu.Allocate<float>((2 * width + 1) * framewidth * frameheight);

            // Run kernel.

            gpu.Launch(new dim3(framewidth, frameheight), 1).medianKernel(width, device_window, depth, device_data, device_meanVector, device_result);
            
            

            // Collect results.
            int[] result = new int[N];
            gpu.CopyFromDevice(device_result, result);
            gpu.Launch(new dim3(framewidth, frameheight), 1).medianKernelInterpolate(width, device_window, depth, device_data, device_meanVector, 3, device_result);

            int[] result2 = new int[N];
            gpu.CopyFromDevice(device_result, result2);
            for (int h = 80; h < 100; h++ )
            {
//                Console.Out.WriteLine((result[h] - result2[h]));
                Console.Out.WriteLine((result[h]));
            }
                //Profile:
                watch.Stop();
            Console.WriteLine("compute time: " + watch.ElapsedMilliseconds);

            // Check some outputs:
            //   Console.WriteLine("Start: " + result[0]);
            
    /*        int start =0*depth;
    
        for (int i = start; i < 102+start; i++)
            {
                Console.WriteLine("Row# " + (i + 1 -start) + ": " + result[i] + " vs " + answer[i-start]);
            }
        */    

            // Clear gpu.
            gpu.FreeAll();
            gpu.HostFreeAll();
            Console.ReadKey(); // keep console up.

        }

        [Cudafy]
        /*
         * kernel takes median width, swapvector that is (2*window_width+1)*x_width*y_height, z_depth, vector with input data organized as:
         *  x1y1z1, x1y1z2,...,x1y1zn, x2y1z1,...
         *  mean value vector for each frame (z1, z2,...)
         *  output vector of same size as input data vector.
         *  
         *  Output is: Input value - frameMean*median.
         *
         */
        public static void medianKernel(GThread thread, int windowWidth, float[] filterWindow, int depth, float[] inputVector, float[] meanVector, int[] answer)
        {
            //          int y   = thread.blockIdx.x;
            //          int x   = thread.threadIdx.x;
            //          int idx = x + (y * thread.blockDim.x);          // which pixel.     

            int y = thread.blockIdx.y;
            int x = thread.blockIdx.x;
            int idx = x + (y * thread.gridDim.x);          // which pixel.  
           
           if (idx < inputVector.Length / depth)           // if pixel is included.
            {
                int zSlice = 0;            // Used to keep track of zslice, used in generating final result values.
                int inputIndex = idx * depth; // Where to start placing results.                
                int filterIndex = idx * (2 * windowWidth + 1); // keep track of filter window insertion.
                Boolean isOdd = true;            // Keep track of if current effective filter window is odd or even, for main portion this is always odd.
                float temp;       // Swap variable for shifting filterWindow entries.
                int answerIndex = idx; // where to put the calclated answer. Output in frame by frame.
                // Start populating filterWindow with windowWidth first number of values from inputVector:
                while (inputIndex < (idx + 1) * depth)
                {
                    inputVector[inputIndex] = inputVector[inputIndex] / meanVector[zSlice];
                    zSlice++;
                    inputIndex++;
                }
                zSlice = 0; // reset.
              //  answerIndex = idx * depth; // reset.
                inputIndex = idx * depth; // reset.
               for (int populateIDX = idx * depth; populateIDX < idx * depth + windowWidth+1; populateIDX++)
                {
                    filterWindow[filterIndex] = inputVector[populateIDX];                    
                    filterIndex++;
                }
                
                // using bubblesort, sort the windowWidth first number of elements:

                int index = idx * (2 * windowWidth + 1); // start point
                
                for (int bubbleSortOuter = index + 1; bubbleSortOuter < index + windowWidth + 1; bubbleSortOuter++)
                {
                    for (int bubbleSortInner = index; bubbleSortInner < index + windowWidth - bubbleSortOuter; bubbleSortInner++)
                    {
                        if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                        {
                            temp = filterWindow[bubbleSortInner];
                            filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                            filterWindow[bubbleSortInner + 1] = temp;
                        }
                    }
                }
                
                if (windowWidth % 2 == 0) // is odd?
                {
                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index + (windowWidth) / 2])); // InputVector - meanvalueFrame*median.
                    if (answer[answerIndex] < 0)
                        answer[answerIndex] = 0;
                    isOdd = false;
                }
                else
                {
                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index + (windowWidth - 1) / 2] + filterWindow[index + ((windowWidth - 1) / 2) + 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                   if (answer[answerIndex] < 0)
                       answer[answerIndex] = 0;
                    isOdd = true;
                }

                // Update counters:
                zSlice++; // index for meanVector.
                if (isOdd)
                    index = idx * (2 * windowWidth + 1) + (windowWidth - 1) / 2 + 1;  // start index for filterWindow.
                else
                    index = idx * (2 * windowWidth + 1) + windowWidth/2 + 1;  // start index for filterWindow.
                inputIndex++; // Index of entry to calculate median for.
                answerIndex += inputVector.Length / depth;
                //answerIndex++;
                for (int populateIndex = idx * depth + windowWidth+1; populateIndex < idx * depth + 2 * windowWidth+1; populateIndex++) // Add one element at a time until 2xwindowWidth+1 elements are in the list.
                {
                    filterWindow[filterIndex] = inputVector[populateIndex];
                    // Bubblesort filterWindow, not pretty but easy to implement:
                    for (int bubbleSortOuter = idx * (2 * windowWidth + 1); bubbleSortOuter < filterIndex; bubbleSortOuter++) // Loop over current filterWindow.
                    {
                        for (int bubbleSortInner = idx * (2 * windowWidth + 1); bubbleSortInner < filterIndex - bubbleSortOuter; bubbleSortInner++)
                        {
                            if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                            {
                                temp = filterWindow[bubbleSortInner];
                                filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                                filterWindow[bubbleSortInner + 1] = temp;
                            }
                        }
                    }

                    if (isOdd) // is odd?
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd = false;                       
                        index++;        // start index for filterWindow.
                    }
                    else
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index] + filterWindow[index - 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;                
                        isOdd = true;
                    }

                    // Update counters:
                    zSlice++;       // index for meanVector.
                    filterIndex++;  // Insertion index for filterWindow.                    
                    inputIndex++; // Index of entry to calculate median for.
                    answerIndex += inputVector.Length / depth;
                    //answerIndex++;
                }
         

                // filterWindow now fully populated.
                // find oldest entry and replace with newest. run through sorting algorithm on this entry and upwards.
                // Loop through centre portion of input data, changing algorithm once end has been reached.                
                filterIndex = idx * (2 * windowWidth + 1); // start index for this coordinate.
                index = filterIndex + windowWidth; // Median element.          
                int upperWindowBound = index + windowWidth; // correct for filter window.              
                Boolean found = false;
                int searchCounter = 0;
                while (inputIndex < (idx + 1) * depth - windowWidth) // until end part has been reached:
                {
                    // find oldest entry and replace with newest.
                    found           = false;           // Update success counter.
                    searchCounter   = filterIndex; // starting index to search in filterWindow.
                    while (!found && searchCounter < upperWindowBound+1)
                    {
                        if (filterWindow[searchCounter] == inputVector[inputIndex - windowWidth - 1]) // find entry in filterWindow matching oldest entry.
                        {
                            found = true;
                            filterWindow[searchCounter] = inputVector[inputIndex + windowWidth]; // replace oldest entry with the next one in line.

                            // Shift filterWindow so that it is ordered again. TODO: Check which order occurs most often. 
                            if (searchCounter == filterIndex && filterWindow[searchCounter] < filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the smallest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] > filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the largest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == filterIndex && filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the larger then the second.
                            {
                                
                                while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                {
                                    temp = filterWindow[searchCounter + 1];
                                    filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter++;
                                }
                                
                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] < filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the smaller then the second last.
                            {
                                
                                while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                {
                                    temp = filterWindow[searchCounter - 1];
                                    filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter--;
                                }
                            } 
                            else if (filterWindow[searchCounter] == filterWindow[searchCounter + 1] || filterWindow[searchCounter] == filterWindow[searchCounter - 1]) // if we just placed a new element equal to one of its neighbours.
                            {
                                // Do nothing, list sorted.
                            }
                            else 
                            {
                                if (filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // if we should shift the new entry upwards.
                                {                                  
                                    while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                    {
                                        temp = filterWindow[searchCounter + 1];
                                        filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter++;
                                    }
                                }
                                else // shift new element down.
                                {
                                    
                                    while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                    {
                                        temp = filterWindow[searchCounter - 1];
                                        filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter--;
                                    }

                                }
                                
                            }

                        } // If correct value has been found.
                        searchCounter++;
                    } // searching while loop.                

                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                    if (answer[answerIndex] < 0)
                        answer[answerIndex] = 0;              
                    zSlice++;       // index for meanVector.                    
                    inputIndex++; // Index of entry to calculate median for.
                    answerIndex += inputVector.Length / depth;
                    //answerIndex++;
                }// main while loop.

                isOdd = false;  // full filter window is always odd number long (2W+1), we start by reducing index by 1.

                int maxFilterIdx = index + windowWidth; // last index for current filterwindow.
          
                while (inputIndex < (idx + 1) * depth) // loop over remaining entries.
                {      
                    if (isOdd)
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd               = false;
                    }
                    else
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index] + filterWindow[index + 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd               = true;
                        index++;
                    }
                    // Update counters:
                    zSlice++;       // index for meanVector.                    
                    inputIndex++;  // Index of entry to calculate median for.                        
                    answerIndex += inputVector.Length / depth;
                    //answerIndex++;
                } // loop over remaining entries.
               // temp load test.
            } // end main check of idx.
        } // End medianKernel.

        [Cudafy]
        /*
         * Skipping nSteps for speedup. Accurate for steps up to 10.
         */ 

        public static void medianKernelInterpolate(GThread thread, int windowWidth, float[] filterWindow, int depth, float[] inputVector, float[] meanVector, int nStep, int[] answer)
        {
            int y = thread.blockIdx.y;
            int x = thread.blockIdx.x;
            int idx = x + (y * thread.gridDim.x);          // which pixel.  

            if (idx < inputVector.Length / depth)           // if pixel is included.
            {
                int zSlice = 0;            // Used to keep track of zslice, used in generating final result values.
                int inputIndex = idx * depth; // Where to start placing results.                
                int filterIndex = idx * (2 * windowWidth + 1); // keep track of filter window insertion.
                Boolean isOdd = true;            // Keep track of if current effective filter window is odd or even, for main portion this is always odd.
                float temp;       // Swap variable for shifting filterWindow entries.
                int answerIndex = idx; // where to put the calclated answer. Output in frame by frame.
                // Start populating filterWindow with windowWidth first number of values from inputVector:
                while (inputIndex < (idx + 1) * depth)
                {
                    inputVector[inputIndex] = inputVector[inputIndex] / meanVector[zSlice];
                    zSlice++;
                    inputIndex++;
                }
                zSlice = 0; // reset.                
                inputIndex = idx * depth; // reset.
                for (int populateIDX = idx * depth; populateIDX < idx * depth + windowWidth + 1; populateIDX += nStep) // load ever nStep entry.
                {
                    filterWindow[filterIndex] = inputVector[populateIDX];
                    filterIndex++;
                }

                // using bubblesort, sort the windowWidth first number of elements:

                int index = idx * (2 * windowWidth + 1); // start point

                for (int bubbleSortOuter = index + 1; bubbleSortOuter < index + windowWidth + 1; bubbleSortOuter++)
                {
                    for (int bubbleSortInner = index; bubbleSortInner < index + windowWidth - bubbleSortOuter; bubbleSortInner++)
                    {
                        if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                        {
                            temp = filterWindow[bubbleSortInner];
                            filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                            filterWindow[bubbleSortInner + 1] = temp;
                        }
                    }
                }

                if (windowWidth % 2 == 0) // is odd?
                {
                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index + (windowWidth) / 2])); // InputVector - meanvalueFrame*median.
                    if (answer[answerIndex] < 0)
                        answer[answerIndex] = 0;
                    isOdd = false;
                }
                else
                {
                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index + (windowWidth - 1) / 2] + filterWindow[index + ((windowWidth - 1) / 2) + 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                    if (answer[answerIndex] < 0)
                        answer[answerIndex] = 0;
                    isOdd = true;
                }

                // Update counters:
                zSlice+=nStep; // index for meanVector.
                if (isOdd)
                    index = idx * (2 * windowWidth + 1) + (windowWidth - 1) / 2 + 1;  // start index for filterWindow.
                else
                    index = idx * (2 * windowWidth + 1) + windowWidth / 2 + 1;  // start index for filterWindow.
                inputIndex+=nStep; // Index of entry to calculate median for.
                answerIndex += (inputVector.Length / depth) * nStep; // next answer is nStep positions away.

                for (int populateIndex = idx * depth + windowWidth + 1; populateIndex < idx * depth + 2 * windowWidth + 1; populateIndex+=nStep) // Add one element at a time until 2xwindowWidth+1 elements are in the list.
                {
                    filterWindow[filterIndex] = inputVector[populateIndex];
                    // Bubblesort filterWindow, not pretty but easy to implement:
                    for (int bubbleSortOuter = idx * (2 * windowWidth + 1); bubbleSortOuter < filterIndex; bubbleSortOuter++) // Loop over current filterWindow.
                    {
                        for (int bubbleSortInner = idx * (2 * windowWidth + 1); bubbleSortInner < filterIndex - bubbleSortOuter; bubbleSortInner++)
                        {
                            if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                            {
                                temp = filterWindow[bubbleSortInner];
                                filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                                filterWindow[bubbleSortInner + 1] = temp;
                            }
                        }
                    }

                    if (isOdd) // is odd?
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd = false;
                        index++;        // start index for filterWindow.
                    }
                    else
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index] + filterWindow[index - 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd = true;
                    }
                    // Interpolate between current and previous point.
                    float step = answer[answerIndex] - answer[answerIndex - (inputVector.Length / depth) * nStep];
                    for (int idx2 = 1; idx2 < nStep; idx2++ )
                    {
                        answer[answerIndex + (inputVector.Length / depth) * (nStep + idx2)] = (int) (answer[answerIndex - (inputVector.Length / depth) * nStep] + idx2 * step);
                    }

                        // Update counters:
                    zSlice += nStep;       // index for meanVector.
                    filterIndex += nStep;  // Insertion index for filterWindow.                    
                    inputIndex += nStep; // Index of entry to calculate median for.
                    answerIndex += (inputVector.Length / depth)*nStep;
                    
                    //answerIndex++;
                }


                // filterWindow now fully populated.
                // find oldest entry and replace with newest. run through sorting algorithm on this entry and upwards.
                // Loop through centre portion of input data, changing algorithm once end has been reached.                
                filterIndex = idx * (2 * windowWidth + 1); // start index for this coordinate.
                index = filterIndex + windowWidth; // Median element.          
                int upperWindowBound = index + windowWidth; // correct for filter window.              
                Boolean found = false;
                int searchCounter = 0;
                while (inputIndex < (idx + 1) * depth - windowWidth) // until end part has been reached:
                {
                    // find oldest entry and replace with newest.
                    found = false;           // Update success counter.
                    searchCounter = filterIndex; // starting index to search in filterWindow.
                    while (!found && searchCounter < upperWindowBound + 1)
                    {
                        if (filterWindow[searchCounter] == inputVector[inputIndex - windowWidth - 1]) // find entry in filterWindow matching oldest entry.
                        {
                            found = true;
                            filterWindow[searchCounter] = inputVector[inputIndex + windowWidth]; // replace oldest entry with the next one in line.

                            // Shift filterWindow so that it is ordered again. TODO: Check which order occurs most often. 
                            if (searchCounter == filterIndex && filterWindow[searchCounter] < filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the smallest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] > filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the largest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == filterIndex && filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the larger then the second.
                            {

                                while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                {
                                    temp = filterWindow[searchCounter + 1];
                                    filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter++;
                                }

                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] < filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the smaller then the second last.
                            {

                                while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                {
                                    temp = filterWindow[searchCounter - 1];
                                    filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter--;
                                }
                            }
                            else if (filterWindow[searchCounter] == filterWindow[searchCounter + 1] || filterWindow[searchCounter] == filterWindow[searchCounter - 1]) // if we just placed a new element equal to one of its neighbours.
                            {
                                // Do nothing, list sorted.
                            }
                            else
                            {
                                if (filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // if we should shift the new entry upwards.
                                {
                                    while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                    {
                                        temp = filterWindow[searchCounter + 1];
                                        filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter++;
                                    }
                                }
                                else // shift new element down.
                                {

                                    while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                    {
                                        temp = filterWindow[searchCounter - 1];
                                        filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter--;
                                    }

                                }

                            }

                        } // If correct value has been found.
                        searchCounter++;
                    } // searching while loop.                

                    answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                    if (answer[answerIndex] < 0)
                        answer[answerIndex] = 0;
                    // Interpolate between current and previous point.
                    float step = answer[answerIndex] - answer[answerIndex - (inputVector.Length / depth) * nStep];
                    for (int idx2 = 1; idx2 < nStep; idx2++)
                    {
                        answer[answerIndex + (inputVector.Length / depth) * (nStep + idx2)] = (int)(answer[answerIndex - (inputVector.Length / depth) * nStep] + idx2 * step);
                    }

                    // Update counters:
                    zSlice += nStep;       // index for meanVector.
                    filterIndex += nStep;  // Insertion index for filterWindow.                    
                    inputIndex += nStep; // Index of entry to calculate median for.
                    answerIndex += (inputVector.Length / depth) * nStep;
                    
                    //answerIndex++;
                }// main while loop.

                isOdd = false;  // full filter window is always odd number long (2W+1), we start by reducing index by 1.

                int maxFilterIdx = index + windowWidth; // last index for current filterwindow.

                while (inputIndex < (idx + 1) * depth) // loop over remaining entries.
                {
                    if (isOdd)
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd = false;
                    }
                    else
                    {
                        answer[answerIndex] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index] + filterWindow[index + 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                        if (answer[answerIndex] < 0)
                            answer[answerIndex] = 0;
                        isOdd = true;
                        index++;
                    }
                    // Interpolate between current and previous point.
                    float step = answer[answerIndex] - answer[answerIndex - (inputVector.Length / depth) * nStep];
                    for (int idx2 = 1; idx2 < nStep; idx2++)
                    {
                        answer[answerIndex + (inputVector.Length / depth) * (nStep + idx2)] = (int)(answer[answerIndex - (inputVector.Length / depth) * nStep] + idx2 * step);
                    }

                    // Update counters:
                    zSlice += nStep;       // index for meanVector.
                    filterIndex += nStep;  // Insertion index for filterWindow.                    
                    inputIndex += nStep; // Index of entry to calculate median for.
                    answerIndex += (inputVector.Length / depth) * nStep;
                    
                    //answerIndex++;
                } // loop over remaining entries.
                answerIndex -= 2*(inputVector.Length / depth) * nStep;
                inputIndex -= nStep;
                zSlice -= nStep;
                
                if (inputIndex != (idx + 1) * depth - 1) // if we did not include the very last entry.
                {
                    if (isOdd)
                    {
                        answer[idx + (depth - 1) * (inputVector.Length / depth)] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - filterWindow[index])); // InputVector - meanvalueFrame*median.
                        if (answer[((idx + 1) * depth)-1] < 0)
                            answer[((idx + 1) * depth)-1] = 0;
                    }
                    else
                    {
                        answer[idx + (depth - 1) * (inputVector.Length / depth)] = (int)(meanVector[zSlice] * (inputVector[inputIndex] - (filterWindow[index] + filterWindow[index + 1]) / 2.0)); // InputVector - meanvalueFrame*median.
                        if (answer[((idx + 1) * depth)-1] < 0)
                            answer[((idx + 1) * depth)-1] = 0;
                    }
                    float step = answer[idx + (depth - 1) * (inputVector.Length / depth)] - answer[answerIndex];
                    for (int idx2 = 1; idx2 < (idx + 1) * depth - 1 - inputIndex; idx2++) 
                    {
                        answer[answerIndex + (inputVector.Length / depth) * idx2] = (int)(answer[answerIndex] + idx2 * step);
                    }
                }
                

            } // end main check of idx.
        } // End medianKernel.
   
        public static float[] generateTest(int N) // generate test vector to simulate mean frame values.
        {
            float[] test = new float[N];
            double[] test2 = { 8434.644, 7944.789, 8194.27, 8411.557, 8423.571, 8768.733, 9151.585, 9589.163, 9292.388, 9192.54, 9241.785, 9390.962, 9120.733, 8684.332, 8567.488, 8683.114, 9384.193, 8444.152, 8558.935, 8188.36, 8265.785, 8175.045, 8652.152, 9035.101, 9499.917, 9521.813, 9455.612, 8975.848, 9469.37, 9262.547, 9419.294, 9250.242, 8780.609, 9055.391, 9711.184, 9466.076, 9089.024, 9320.706, 9524.276, 9299.751, 8867.488, 8761.397, 8585.647, 8731.46, 8768.208, 9142.298, 9145.453, 9735.944, 9030.727, 8791.737, 8122.2974, 8496.457, 8544.955, 9327.405, 9060.886, 8238.782, 8119.571, 8147.5986, 8796.789, 9054.229, 9454.104, 9465.439, 9805.26, 9646.173, 9739.695, 9378.921, 8788.152, 9122.823, 8830.422, 8800.18, 10060.305, 10165.868, 9758.921, 9544.996, 9449.426, 9405.536, 9266.519, 9135.861, 9600.484, 9618.893, 9180.567, 9310.104, 9426.727, 9248.692, 9276.65, 8808.567, 8687.82, 8480.374, 8745.896, 9121.287, 9010.007, 9159.973, 9398.879, 9205.8545, 8743.28, 9023.848, 8256.747, 7917.3843, 8099.1143, 8149.2734, 8684.595, 8410.616, 8302.422, 8018.7407, 7841.8545, 8530.505, 8942.312, 8624.415, 9213.426, 9033.592, 8886.159, 8934.353, 8689.827, 9059.986, 9015.433, 8722.491, 9000.762, 8985.605, 8810.27, 8719.474, 8308.027, 8298.823, 8412.249, 8083.211, 8121.9375, 8284.595, 8623.931, 8294.284, 7954.713, 8188.692, 8279.28, 8800.996, 8660.845, 8735.28, 8921.467, 8561.467, 8416.941, 8292.125, 8555.848, 8501.343, 8653.356, 8966.0205, 8893.827, 9039.28, 8644.014, 8390.686, 8548.097, 8979.807, 9128.097, 8759.903, 8597.73, 8543.405, 8322.215, 8515.419, 8716.609, 8877.688, 9180.359, 9083.253, 9386.104, 8967.585, 8710.256, 8480.166, 8077.2456, 8036.9136, 8361.785, 8499.751, 8685.329, 8565.453, 8587.778, 9097.177, 8906.464, 9017.883, 8529.231, 8670.782, 8616.318, 8025.4116, 7972.7197, 7949.5776, 8197.9375, 8691.101, 8560.097, 8880.623, 9293.481, 8914.769, 8297.052, 8575.336, 8617.038, 8082.6987, 8573.661, 8042.464, 8414.906, 8525.384, 9222.491, 9067.322, 8774.935, 8594.464, 9121.024, 8881.951, 8407.238, 8297.079, 8206.769, 8477.633, 8226.727, 8217.26, 8303.128, 8176.346, 7890.3667, 7768.263, 7628.5674, 7372.0, 7749.1904, 7832.5815, 8090.547, 8713.509, 8835.641, 8998.0625, 9591.724, 9084.747, 9646.45, 9259.142, 9127.267, 9134.325, 8918.962, 8920.982, 8360.637, 8615.724, 8247.253, 8057.841, 8559.46, 9012.291, 8990.0625, 9101.924, 8707.045, 8968.346, 8178.7407, 7853.8687, 7596.111, 7992.858, 8103.4185, 8094.768, 8418.782, 8326.339, 8574.603, 9061.9375, 8984.609, 8968.318, 8609.675, 8153.564, 8405.924, 8375.903, 8176.263, 8912.512, 8483.958, 8591.488, 8709.758, 8618.948, 8678.796, 8657.024, 8566.0205, 8602.574, 8409.578, 8404.637, 7948.415, 7808.8994, 7808.3735, 8213.495, 8527.267, 8941.744, 9405.813, 9471.35, 9062.491, 8384.11, 7930.021, 8267.17, 7875.806, 7552.028, 8646.962, 8338.603, 8113.7715, 8173.2593, 8055.9033, 7976.8164, 7671.4463, 7328.8027, 7480.346, 7431.377, 7356.249, 7690.3945, 7871.6953, 8007.3496, 8478.491, 8553.26, 8584.249, 8801.883, 8450.436, 8844.789, 8587.737, 8316.042, 8279.253, 8213.273, 8601.121, 8761.343, 9099.641, 9131.792, 9245.951, 9485.966, 9135.502, 8754.242, 8491.765, 8485.467, 8512.249, 8269.688, 8406.132, 8678.325, 9034.671, 9459.792, 8984.429, 8288.18, 8727.543, 8260.222, 8755.571, 8865.135, 8479.931, 8383.031, 8024.429, 8057.661, 7589.149, 7213.924, 7460.36, 7760.955, 7853.4395, 8435.682, 8340.443, 8001.2456, 7985.343, 7778.1455, 8100.7334, 8175.862, 8097.6055, 7892.9272, 7804.4565, 7990.9067, 8623.612, 8454.45, 8500.733, 8185.896, 8126.7544, 8436.401, 8463.516, 8726.464, 8778.769, 9197.619, 8960.042, 9021.799, 8413.079, 8207.17, 8014.353, 8083.737, 7820.138, 7966.5605, 8124.678, 7879.1973, 8117.453, 8153.6885, 7811.723, 7762.1455, 7904.6367, 8003.488, 8309.868, 8755.557, 8112.512, 8870.0205, 8503.89, 8742.076, 8841.9375, 8724.388, 8649.509, 8715.018, 8890.686, 8911.986, 8714.588, 8731.931, 8140.443, 7790.685, 7557.896, 7810.5327, 7872.6504, 8009.924, 8184.609, 8036.2075, 8401.356, 8470.367, 8502.395, 8527.35, 8147.5986, 7930.159, 7815.8203, 7683.225, 7696.138, 8100.9414, 8004.8994, 8246.09, 7990.6436, 7219.0586, 7216.5674, 7391.156, 7471.239, 7528.1523, 7508.692, 7165.343, 7267.9307, 7183.5293, 7143.4326, 7529.896, 7748.6367, 7541.5776, 7556.678, 7395.045, 7464.028, 7410.339, 7203.862, 7614.6157, 7582.3667, 7399.751, 7013.536, 7172.7197, 7312.2217, 7699.889, 7764.3184, 7685.038, 7952.194, 7709.0244, 7225.73, 7401.9517, 7848.346, 7925.3564, 8122.1313, 8203.516, 8035.017, 8227.142, 8094.1313, 8124.706, 7638.934, 7669.7715, 7811.3354, 8153.6333, 7969.536, 8083.5293, 8523.488, 8033.896, 7581.9517, 7618.7266, 7799.5293, 7717.0103, 7272.2217, 7651.668, 7546.187, 7942.7266, 8149.052, 8053.37, 8120.028, 7924.512, 7407.5435, 7078.3115, 7076.249, 6843.9033, 7321.398, 7625.5503, 7462.934, 7056.761, 7313.7163, 7540.5396, 8097.841, 8221.564, 8198.339, 7766.8374, 7659.5156, 7276.0137, 6895.6265, 6993.398, 7333.232, 7473.9653, 7445.4395, 7259.4326, 7405.315, 7193.7715, 7052.028, 7300.332, 6943.7925, 7329.5225, 7256.498, 7259.5986, 7263.4185, 7405.135, 7950.8926, 8156.4565, 8501.924, 8052.0693, 7767.017, 7237.619, 7606.3667, 7927.2803, 7711.972, 7636.0, 7765.7026, 7703.6816, 7401.0933, 7210.1313, 7364.9272, 7370.2837, 7177.896, 6681.426, 6719.9863, 6933.5503, 7203.9585, 7371.128, 6754.8237, 7032.0415, 7091.723, 7288.1523, 7372.4014, 7660.9136, 7592.36, 7676.0967, 8044.498, 8259.516, 8579.571, 8573.37, 8315.737, 8123.889, 7878.9756, 7503.5986, 7527.0864, 7441.8687, 7847.585, 8144.429, 8297.329, 8647.82, 8588.789, 8393.094, 7949.066, 8290.823, 8191.889, 7954.63, 7658.2974, 7559.5986, 8052.5396, 8087.6123, 7419.557, 7618.9756, 8023.668, 7770.7544, 7826.7266, 7628.443, 7151.834, 7241.0933, 7577.9517, 7648.6646, 7505.509, 7481.841, 7750.768, 7571.8477, 7712.166, 7852.678, 7783.557, 7526.021, 7710.547, 7967.806, 7763.322, 7460.332, 7662.547, 7313.924, 7558.339, 7336.3735, 7498.934, 7731.806, 8029.3843, 7737.287, 7947.3633, 7798.104, 7285.9653, 7474.768, 7803.557, 7996.0693, 7646.8096, 7624.8857, 7840.4014, 8002.3945, 8223.792, 7905.2456, 7901.7715, 7925.7715, 8173.5776, 7685.2456, 7334.2837, 7058.796, 7407.7925, 7270.0483, 7068.5396, 7626.865, 7990.1455, 7760.0693, 7508.332, 7618.9756, 7869.2593, 8245.163, 7664.872, 7678.187, 7501.979, 7765.744, 7831.502, 7593.2456, 7427.4604, 7560.872, 7947.2524, 7637.9517, 7951.8203, 8137.979, 7987.972, 7137.315, 7412.4844, 7654.574, 7324.512, 7214.4497, 8127.806, 7920.263, 7862.242, 7949.135, 8258.657, 8656.928, 8376.692, 8026.339, 7733.038, 8186.9067, 7603.585, 8206.533, 8305.536, 8324.18, 8176.498, 7943.571, 7735.571, 7801.619, 8151.474, 7636.498, 7931.5156, 8055.4185, 7777.453, 7691.7095, 7879.8755, 8066.713, 8195.848, 7950.0625, 7781.3843, 6996.498, 7435.8755, 7516.7334, 7422.0347, 7452.872, 7457.315, 7098.436, 7434.782, 7374.865, 7882.768, 8093.993, 8172.111, 8138.408, 7672.0693, 7430.173, 7577.3286, 7666.6714, 7780.429, 7814.962, 8046.007, 8201.356, 8139.945, 8364.208, 8571.751, 8777.163, 8388.581, 8288.346, 8166.353, 8872.291, 8649.951, 8431.765, 8283.46, 8287.474, 8580.235, 7974.9204, 7627.0728, 7917.2734, 8229.522, 7710.256, 8192.18, 7771.5293, 7269.135, 7696.761, 7458.8374, 7208.872, 7516.983, 7449.0933, 7367.5986, 7334.381, 7188.4565, 7527.5293, 7797.4116, 7694.1313, 7558.3667, 7634.8374, 7666.2285, 7591.1694, 7295.1973, 7349.038, 7294.104, 7003.64, 7057.0103, 7552.3735, 8000.3877, 7553.592, 7797.9653, 7911.806, 8632.581, 7870.63, 8011.5435, 8133.343, 7930.2974, 7440.0, 7063.0034, 7006.9897, 7268.789, 7345.6333, 6855.0586, 7022.934, 7474.2974, 7755.239, 7629.924, 7281.218, 7470.3945, 7365.8823, 7352.055, 7400.775, 7596.166, 7930.8237, 7518.09, 7684.872, 7232.512, 7621.121, 7760.512, 7638.242, 7187.6816, 7076.4014, 6822.436, 6857.813, 6998.6987, 6558.076, 6742.602, 6693.91, 6465.0796, 6369.91, 6843.128, 6987.474, 7115.1973, 7150.436, 8108.4844, 7900.8857, 7417.3564, 7460.6646, 7336.6504, 7308.678, 7221.6333, 7260.3047, 7331.211, 7310.1455, 7339.9863, 6989.9653, 6916.4707, 6846.27, 7032.512, 7327.9585, 7378.325, 7346.5327, 7057.218, 7233.2456, 6871.142, 7072.706, 7007.2666, 6813.7993, 6801.7163, 6611.6265, 6552.36, 6811.1836, 6849.619, 6839.4463, 7073.0103, 7584.3184, 7236.512, 7442.408, 7583.6953, 7225.2734, 6603.1143, 6264.983, 6531.6265, 6410.782, 6425.6885, 6779.405, 7062.2837, 6861.7856, 7108.844, 6721.121, 6609.1904, 6897.343, 7302.9897, 7391.4185, 7723.6816, 7612.8027, 7374.3115, 7509.315, 7416.263, 7287.0312, 7258.4497, 7057.3013, 7334.007, 7512.36, 7695.889, 7922.8374, 8042.339, 7831.8755, 7766.464, 7679.2666, 7728.263, 7401.038, 7136.3877, 6664.844, 6640.7197, 6447.5986, 6422.27, 6779.017, 6830.381, 6764.166, 7129.149, 7161.5225, 6978.7266, 7161.135, 6805.1074, 6925.7993, 6892.858, 6811.6123, 7000.6367, 6664.5815, 6755.6123, 6429.758, 6358.685, 6748.7476, 6537.426, 6586.0625, 6630.6987, 6716.2905, 7299.2666, 7499.6123, 7521.7163, 7886.5327, 7951.6816, 8019.737, 7582.9204, 7538.339, 7250.8926, 7195.9585, 7296.6646, 6908.4844, 6848.775, 6789.2593, 5992.1245, 6148.2905, 6447.6953, 6777.4395, 6801.3286, 6324.983, 6728.0415, 6647.4185, 6532.3184, 6933.3013, 7072.3877, 6812.7334, 7184.6646, 6841.8687, 6615.5986, 6794.5054, 7249.0244, 7069.6333, 7079.945, 7119.1143, 7106.962, 6997.91, 7201.232, 7485.0796, 7020.692, 6651.2524, 6980.9272, 7453.218, 7832.194, 7740.9272, 7182.187, 7003.6123, 7157.661, 7038.865, 7284.9414, 7219.7095, 7032.4844, 7288.6646, 7221.052, 6918.962, 6869.0933, 7413.4395, 7880.9688, 8080.4565, 7834.3115, 7620.8994, 7737.1626, 7247.322, 7465.121, 7056.3047, 7503.1143, 7125.052, 6710.962, 6437.1904, 6514.8926, 6442.173, 6215.806, 6287.308, 6274.7407, 6767.239, 6765.0796, 6382.865, 6636.2354, 7367.211, 7078.602, 7523.6123, 7889.121, 8021.5503, 7904.706, 7292.055, 7268.083, 7595.7095, 7465.896, 7493.7715, 8087.654, 7540.8857, 7214.09, 7434.09, 7654.0347, 7184.4844, 7029.661, 7422.2144, 6995.6123, 7016.1797, 7166.408, 7209.135, 7142.934, 7487.3633, 7665.0244, 7272.8857, 6938.8096, 6978.325, 7083.488, 7271.239, 7274.8374, 7386.934, 7527.1836, 7311.4463, 7053.9517, 7245.343, 7374.187, 7721.7993, 8122.0483, 8056.138, 7885.398, 7953.2734, 8098.685, 8091.7095, 7924.872, 7707.2524, 7379.156, 7255.7925, 7172.512, 6840.692, 6564.249, 6536.6367, 6665.647, 6768.1245, 7010.3667, 7309.2734, 7320.1797, 7104.6367, 7090.4775, 6667.9307, 6743.862, 6790.962, 6640.5537, 6566.325, 6621.038, 6560.4565, 6648.0137 };
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = (float)test2[i];
            }
            return test;
        }

        public static float[] generateTest(int N, float depth) // Generate test vector to easily see if median calculations are correct.
        {
            float[] test = new float[N];
            double[] test2 = { 13984, 8432, 7808, 8016, 8032, 9216, 8832, 9424, 8592, 8256, 9216, 9760, 9312, 9344, 8624, 9888, 8912, 6928, 8000, 8016, 8944, 6928, 9472, 9056, 8928, 9216, 7328, 8128, 8080, 7504, 7872, 7744, 7840, 8016, 9264, 9136, 10272, 9568, 10096, 7776, 9456, 8496, 7152, 10176, 9040, 8256, 9664, 10144, 9632, 9072, 8720, 10064, 9856, 9344, 9008, 8640, 9008, 7888, 8272, 9088, 8128, 7840, 8256, 7984, 7664, 8608, 9232, 8384, 7696, 7696, 7968, 8960, 7584, 8000, 8832, 9328, 8912, 8400, 9184, 9056, 9104, 7632, 9312, 9264, 7328, 7664, 8112, 7296, 8432, 7456, 8224, 8352, 10528, 7376, 8176, 8288, 7712, 7968, 7424, 7280, 7680, 6752, 6288, 8208, 7472, 12288, 9456, 9648, 8752, 8592, 11296, 9968, 13824, 13280, 14832, 14160, 14208, 14400, 10016, 11472, 12224, 15248, 12528, 7840, 7568, 7840, 11904, 11648, 9328, 9136, 7792, 8512, 7744, 7824, 8608, 8000, 8608, 7904, 8912, 8240, 11072, 12592, 11456, 9632, 9600, 7440, 6880, 7440, 7536, 8432, 8064, 7792, 7920, 10944, 12352, 15296, 12304, 12080, 16928, 10352, 6368, 6816, 10688, 7488, 7024, 9104, 8512, 8192, 6896, 7552, 8352, 7056, 8064, 7536, 7056, 8272, 7552, 6800, 6976, 7808, 7872, 8000, 6608, 8784, 8208, 7840, 9808, 8144, 7840, 8416, 8192, 9120, 8128, 9760, 11344, 9616, 9696, 10080, 9248, 8416, 9712, 10240, 8752, 10016, 9520, 9136, 8432, 9200, 8336, 8688, 8880, 9072, 9312, 8816, 10016, 8544, 9056, 8592, 8592, 9168, 12192, 10416, 12016, 8512, 7856, 8720, 7328, 8592, 7728, 9008, 9456, 9360, 8768, 7856, 8368, 8896, 8256, 9248, 7840, 9184, 8912, 9680, 8096, 8720, 8528, 9024, 8112, 9440, 7280, 9600, 7120, 8064, 7200, 5648, 7136, 8352, 6992, 7312, 8080, 9232, 8432, 7840, 9152, 7392, 6416, 7440, 8336, 9088, 10736, 9824, 8768, 8080, 7952, 7904, 8656, 8992, 7584, 6784, 7744, 8032, 8048, 7872, 6736, 7856, 7440, 7568, 8016, 7824, 8928, 7248, 8272, 7296, 6768, 7600, 7248, 8512, 9552, 9408, 10944, 11056, 8656, 6720, 8880, 12576, 8768, 6800, 7472, 10064, 9680, 10736, 7952, 8032, 7344, 7216, 7968, 8304, 8432, 8048, 9104, 8416, 8048, 9248, 8336, 8464, 8048, 7904, 6944, 7872, 7808, 7936, 7168, 7344, 7952, 7936, 8416, 10528, 11680, 7792, 7728, 6896, 7696, 6960, 7200, 7296, 9136, 7840, 6896, 7312, 8992, 7936, 7056, 8720, 7296, 8704, 8000, 7184, 9648, 8880, 6912, 8848, 7584, 8160, 7536, 8352, 7232, 7392, 8464, 7648, 8272, 8016, 8128, 8992, 11344, 10736, 13200, 10112, 7472, 8144, 8400, 7792, 7760, 8944, 8816, 7664, 7744, 7520, 7248, 8016, 7760, 8384, 7856, 7040, 7728, 9008, 7408, 11296, 11232, 13808, 12240, 8720, 7440, 6944, 7440, 7040, 7408, 7568, 6880, 6112, 7216, 9040, 10960, 11088, 9104, 11680, 10272, 11632, 11248, 12048, 11184, 9520, 11216, 9024, 9424, 8048, 7312, 6976, 6736, 6272, 7040, 6832, 6992, 7248, 7120, 6864, 7824, 7504, 7632, 6544, 7744, 7504, 7776, 7328, 7792, 6976, 6720, 7776, 8208, 7264, 8608, 6704, 8960, 6640, 7040, 6336, 5936, 8256, 8160, 8096, 8384, 7312, 7056, 7664, 7712, 6640, 6752, 7312, 7904, 7824, 9408, 8224, 7424, 7296, 6928, 7216, 8368, 7440, 7424, 7024, 6720, 6176, 6144, 6688, 7728, 6784, 7216, 7696, 6416, 6832, 5072, 5936, 7568, 6032, 5760, 6224, 7024, 7696, 7456, 5920, 7040, 6960, 6640, 6576, 7408, 6848, 7920, 7168, 7664, 7872, 6656, 6912, 6736, 6976, 7952, 6864, 6448, 6976, 7424, 8080, 6032, 6240, 7440, 7424, 7360, 9712, 6928, 6960, 8016, 7232, 7568, 6592, 7504, 6960, 6912, 6320, 7264, 6976, 7648, 9520, 8208, 7616, 8624, 7616, 8416, 7264, 8272, 6928, 7552, 9408, 10560, 10416, 11456, 8928, 9920, 10096, 7024, 6688, 6512, 7632, 8192, 7616, 7376, 7104, 8464, 7280, 7200, 6800, 6656, 6608, 8016, 7200, 8080, 7328, 8032, 7312, 7056, 6672, 6512, 7120, 7424, 7248, 7440, 8016, 8544, 8848, 8080, 7888, 6704, 7152, 8544, 9120, 6592, 8480, 8688, 8528, 7824, 6800, 8304, 6720, 7120, 7520, 7776, 5376, 7536, 7760, 8064, 9648, 6496, 7136, 7120, 7344, 6352, 6544, 6304, 7264, 6576, 6256, 7712, 6848, 8640, 7296, 7472, 8400, 7312, 7344, 6928, 7856, 7696, 8336, 8128, 9360, 8176, 6544, 7312, 6912, 6976, 7152, 6304, 7344, 6560, 7760, 7184, 6608, 7552, 7232, 8672, 8336, 7424, 7504, 7600, 8016, 6592, 6224, 6560, 6608, 6432, 6784, 6752, 6400, 8128, 11872, 10256, 8688, 10320, 12048, 10800, 8752, 7296, 7376, 6640, 7072, 7136, 8016, 8416, 7984, 7280, 8272, 8368, 6864, 7536, 7968, 8064, 8416, 8768, 8880, 8384, 8880, 9600, 8672, 9328, 8304, 9328, 8928, 8400, 9664, 10448, 9808, 10048, 7712, 7696, 6400, 6816, 6528, 6784, 6592, 9072, 7984, 7536, 6864, 7136, 6848, 7632, 7088, 5968, 7232, 8160, 6992, 7440, 6752, 7248, 8368, 8096, 7392, 7632, 7872, 6224, 6976, 7472, 6880, 6176, 6608, 6480, 6096, 7504, 7232, 8192, 6544, 6960, 6768, 6128, 7088, 6784, 6928, 7296, 7312, 7200, 6816, 6560, 6768, 5936, 7408, 6000, 6544, 9408, 7552, 7056, 8288, 7280, 5536, 5728, 6064, 7136, 5872, 6288, 7680, 7072, 6688, 6832, 6752, 6944, 7296, 7616, 7200, 7008, 6752, 6832, 6640, 6832, 7136, 6864, 7392, 7984, 8064, 6592, 6864, 7712, 7504, 6544, 6880, 7184, 6448, 6608, 7696, 6976, 7552, 7088, 7056, 6160, 6544, 6944, 6528, 7456, 6624, 6640, 7216, 7888, 7424, 6640, 6848, 5872, 6688, 7024, 6752, 8032, 7696, 7520, 7504, 6864, 7296, 6368, 6384, 6592, 6640, 6976, 6560, 7440, 8032, 7456, 7200, 10896, 11472, 10720, 7760, 6672, 6608, 6960, 6800, 6944, 7120, 8048, 7888, 7952, 7024, 6416, 7776, 7248, 7408, 6976, 6880, 6624, 6208, 6432, 6448, 5984, 6336, 9728, 6992, 6000, 7312, 7376, 7488, 6736, 8128, 6656, 6784, 6672, 6784, 6576, 5600, 6160, 5568, 6048, 7136, 6912, 6656, 8464, 7696, 6192, 6912, 5888, 7168, 7264, 6384, 6416, 6256, 7248, 6800, 7344, 6512, 7200, 7392, 8368, 8176, 7584, 7216, 7696, 8144, 8480, 8448, 8432, 6816, 7840, 7472, 6880, 7184, 7584, 6992, 6720, 6176, 6400, 5760, 7120, 7056, 6752, 8352, 7936, 7408, 7936, 7728, 7136, 7376, 7184, 6400, 7696, 6992, 6848, 7264, 6080, 6576, 5232, 7040, 6352, 6080, 7472, 7264, 6592, 7904, 7440, 7696, 7216, 8096, 7264, 8080, 9104, 8864, 9840, 10752, 7632, 8112, 7280, 7824, 8240, 7504, 8384, 7456, 7264, 8352, 9920, 7728, 7648, 7152, 6080, 6656, 6560, 6816, 6784, 6720, 6224, 6096, 6304, 7616, 6016, 6448, 6640, 7008, 6640, 5984, 7776, 6736, 7520, 7360, 5984, 6768, 5648, 7744, 6048, 5792, 6864, 6208, 5808, 6784, 6240, 5920, 7632, 6576, 5952, 6672, 6272, 8160, 5840, 6464, 5568 };
            int count = 0;
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = (float)test2[count];
                
                count++;
                if (count == test2.Length)
                    count = 0;
            }
            return test;
        }
    }
}
