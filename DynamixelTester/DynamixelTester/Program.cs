using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Text;
using System.Threading;
using dynamixel_sdk;                    // for dynamixel


namespace DynamixelTester
{
    class Program
    {
        // Control table address
        public const int ADDR_MX_TORQUE_ENABLE = 24;                  // Control table address is different in Dynamixel model
        public const int ADDR_MX_GOAL_POSITION = 30;
        public const int ADDR_MX_PRESENT_POSITION = 36;

        // Protocol version
        public const int PROTOCOL_VERSION = 1;                   // See which protocol version is used in the Dynamixel

        // Default setting
        public const int DXL_ID = 3;                   // Dynamixel ID: 1
        public const int BAUDRATE = 1000000;
        public const string DEVICENAME = "COM5";              // Check which port is being used on your controller
                                                              // ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

        public const int TORQUE_ENABLE = 1;                   // Value for enabling the torque
        public const int TORQUE_DISABLE = 0;                   // Value for disabling the torque
        public const int ROT_MIN_POS = 1028;                 // Dynamixel will rotate between this value
        public const int ROT_MAX_POS = 3073;                // and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)

        public const int FLEX_MIN_POS = 790;                 // Dynamixel will rotate between this value
        public const int FLEX_MAX_POS = 3328;

        public const int DXL_MOVING_STATUS_THRESHOLD = 10;                  // Dynamixel moving status threshold

        public const byte ESC_ASCII_VALUE = 0x1b;

        public const int COMM_SUCCESS = 0;                   // Communication Success result value
        public const int COMM_TX_FAIL = -1001;               // Communication Tx Failed


        static void Main(string[] args)
        {
            Stopwatch stopWatch1 = new Stopwatch();
            long milliSec1;

            // Initialize PortHandler Structs
            // Set the port path
            // Get methods and members of PortHandlerLinux or PortHandlerWindows
            int port_num = dynamixel.portHandler(DEVICENAME);

            StreamWriter rotLog = new StreamWriter("rot_log.txt", false);
            StreamWriter rotFlexUpLog = new StreamWriter("rot_flex_up_log.txt", false);

            StreamWriter flexLog = new StreamWriter("flex_log.txt", false);


            int signalDuration = 5000;
            double[] frequencies = new double[] { 0.5, 1.0, 5.0, 10, 50, 100 };
            int[][] signals = new int[frequencies.Length][];

            for(int i = 0; i < frequencies.Length; i++)
            {
                signals[i] = new int[signalDuration];
                int val = 0;
                double period = 1/ (frequencies[i] / 1000);
                for(int j = 0; j < signalDuration; j++)
                {
                    signals[i][j] = val;
                    if (j % period == 0 && j != 0)
                    {
                        val = val == 0 ? 1 : 0;
                    }
                }
            }



            // Initialize PacketHandler Structs
            dynamixel.packetHandler();

            UInt16 dxl_present_position = 0;                                      // Present position

            // Open port
            if (dynamixel.openPort(port_num))
            {
                Console.WriteLine("Succeeded to open the port!");
            }
            else
            {
                Console.WriteLine("Failed to open the port!");
                Console.WriteLine("Press any key to terminate...");
                Console.ReadKey();
                return;
            }

            // Set port baudrate
            if (dynamixel.setBaudRate(port_num, BAUDRATE))
            {
                Console.WriteLine("Succeeded to change the baudrate!");
            }
            else
            {
                Console.WriteLine("Failed to change the baudrate!");
                Console.WriteLine("Press any key to terminate...");
                Console.ReadKey();
                return;
            }

            // Enable Dynamixel Torque
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE);

            if (!checkResult(port_num))
            {
                Console.WriteLine("Dynamixel has been successfully connected");
            }

            //ushort downVal = ROT_MIN_POS + 500;
            //ushort upVal = ROT_MAX_POS - 500;
            ushort downVal = (ROT_MIN_POS + ROT_MAX_POS) / 2 - 100;
            ushort upVal = (ROT_MIN_POS + ROT_MAX_POS) / 2 + 100;

            rotLog.WriteLine("timestamp,freq,time,sig,pos");

            for (int i = 0; i < frequencies.Length; i++)
            {
                Console.WriteLine("Start Frequency Test: " + frequencies[i] + " Hz");
                resetPos(port_num, 2048, 2048);
                stopWatch1.Reset();
                stopWatch1.Start();
                for (int j = 0; j < signalDuration-4; j+=5)
                {
                    ushort pos = 0;
                    if (signals[i][j] == 0)
                    {
                        pos = downVal;
                    }
                    else
                    {
                        pos = upVal;
                    }
                    dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_GOAL_POSITION, pos);

                    dxl_present_position = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_PRESENT_POSITION);

                    stopWatch1.Stop();
                    milliSec1 = stopWatch1.ElapsedMilliseconds;
                    if (milliSec1 < 5)
                    {
                        Thread.Sleep((int)(5 - milliSec1));
                    }

                    rotLog.WriteLine(string.Format("{0}, {1}, {2}, {3}, {4}", DateTime.Now.ToString("HH:mm:ss.fff"), frequencies[i], j, pos, dxl_present_position ));

                    stopWatch1.Reset();
                    stopWatch1.Start();
                }
            }

            rotFlexUpLog.WriteLine("timestamp,freq,time,sig,pos");

            for (int i = 0; i < frequencies.Length; i++)
            {
                Console.WriteLine("Start Frequency Test: " + frequencies[i] + " Hz");
                resetPos(port_num, 2048, 3328);
                stopWatch1.Reset();
                stopWatch1.Start();
                for (int j = 0; j < signalDuration - 4; j += 5)
                {
                    ushort pos = 0;
                    if (signals[i][j] == 0)
                    {
                        pos = downVal;
                    }
                    else
                    {
                        pos = upVal;
                    }
                    dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_GOAL_POSITION, pos);

                    dxl_present_position = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_PRESENT_POSITION);

                    stopWatch1.Stop();
                    milliSec1 = stopWatch1.ElapsedMilliseconds;
                    if (milliSec1 < 5)
                    {
                        Thread.Sleep((int)(5 - milliSec1));
                    }

                    rotFlexUpLog.WriteLine(string.Format("{0}, {1}, {2}, {3}, {4}", DateTime.Now.ToString("HH:mm:ss.fff"), frequencies[i], j, pos, dxl_present_position));

                    stopWatch1.Reset();
                    stopWatch1.Start();
                }
            }

            //downVal = FLEX_MIN_POS + 500;
            //upVal = FLEX_MAX_POS - 500;
            downVal = (FLEX_MIN_POS + FLEX_MAX_POS) / 2 - 100;
            upVal = (FLEX_MIN_POS + FLEX_MAX_POS) / 2 + 100;

            flexLog.WriteLine("timestamp,freq,time,sig,pos");

            for (int i = 0; i < frequencies.Length; i++)
            {
                Console.WriteLine("Start Frequency Test: " + frequencies[i] + " Hz");
                resetPos(port_num, 2048, 2048);
                stopWatch1.Reset();
                stopWatch1.Start();
                for (int j = 0; j < signalDuration - 4; j += 5)
                {
                    ushort pos = 0;
                    if (signals[i][j] == 0)
                    {
                        pos = downVal;
                    }
                    else
                    {
                        pos = upVal;
                    }
                    dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_GOAL_POSITION, pos);
                    dxl_present_position = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_PRESENT_POSITION);

                    stopWatch1.Stop();
                    milliSec1 = stopWatch1.ElapsedMilliseconds;
                    if (milliSec1 < 5)
                    {
                        Thread.Sleep((int)(5 - milliSec1));
                    }

                    flexLog.WriteLine(string.Format("{0}, {1}, {2}, {3}, {4}", DateTime.Now.ToString("HH:mm:ss.fff"), frequencies[i], j, pos, dxl_present_position));

                    stopWatch1.Reset();
                    stopWatch1.Start();
                }
            }

            // Disable Dynamixel Torque
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
            checkResult(port_num);

            rotLog.Dispose();
            rotFlexUpLog.Dispose();
            flexLog.Dispose();

            // Close port
            dynamixel.closePort(port_num);

            return;
        }
        public static void resetPos(int port_num, ushort rot, ushort flex)
        {
            dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_GOAL_POSITION, rot);
            dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_GOAL_POSITION, flex);
            Thread.Sleep(1000);

        }
        public static bool checkResult(int port_num)
        {
            int dxl_comm_result = COMM_TX_FAIL;                                   // Communication result
            byte dxl_error = 0;                                                   // Dynamixel error


            if ((dxl_comm_result = dynamixel.getLastTxRxResult(port_num, PROTOCOL_VERSION)) != COMM_SUCCESS)
            {
                Console.WriteLine(dxl_comm_result);
                return false;
            }
            else if ((dxl_error = dynamixel.getLastRxPacketError(port_num, PROTOCOL_VERSION)) != 0)
            {
                Console.WriteLine(dxl_error);
                return false;
            }
            return true;
        }
    }
}
