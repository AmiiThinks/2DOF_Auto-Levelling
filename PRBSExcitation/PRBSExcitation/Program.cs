using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Text;
using System.Threading;
using dynamixel_sdk;                    // for dynamixel


namespace APRBSExcitation
{
    class Program
    {
        // Control table address
        public const int ADDR_MX_TORQUE_ENABLE = 24;                  // Control table address is different in Dynamixel model
        public const int ADDR_MX_GOAL_POSITION = 30;
        public const int ADDR_MX_PRESENT_POSITION = 36;
        public const int ADDR_MX_PRESENT_SPEED = 38;

        // Protocol version
        public const int PROTOCOL_VERSION = 1;                   // See which protocol version is used in the Dynamixel

        // Default setting
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

            StreamWriter aprbsLog = new StreamWriter("aprbs_180ptp_log.txt", false);

            List<ushort> command = new List<ushort>();

            using (var reader = new StreamReader(@"C:\Users\James\Documents\Bypass_Prothesis\2DOF_Auto-Levelling\al_python\NN Methods\aprbs.csv"))
            {
                Console.WriteLine("Test");
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    double angle = Convert.ToDouble(line);
                    ushort ticks = (ushort)((angle + 180) * 4096 / 360);
                    command.Add(ticks);
                }
            }


            // Initialize PacketHandler Structs
            dynamixel.packetHandler();


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

            aprbsLog.WriteLine("timestamp,rot_command,flex_command,rot_cur_pos,flex_cur_pos,rot_next_pos,flex_next_pos,rot_cur_vel,flex_cur_vel,rot_next_vel,flex_next_vel");

            UInt16 rotNextPos = 2048;                                     // Present position
            UInt16 flexNextPos = 2048;                                      // Present position
            UInt16 rotCurPos = 0;
            UInt16 flexCurPos = 0;

            UInt16 rotNextVel = 0;                                     // Present velocity
            UInt16 flexNextVel = 0;                                      // Present velocity
            UInt16 rotCurVel = 0;
            UInt16 flexCurVel = 0;

            ushort rotCommand = 0;
            ushort flexCommand = 0;
            string timeStamp = DateTime.Now.ToString("HH:mm:ss.fff");


            resetPos(port_num, 2048, 2048);
            rotCurPos = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_PRESENT_POSITION);
            flexCurPos = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_PRESENT_POSITION);

            stopWatch1.Reset();
            stopWatch1.Start();

            for (int i = 0; i < command.Count; i++)
            {

                rotNextPos = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_PRESENT_POSITION);
                flexNextPos = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_PRESENT_POSITION);

                rotNextVel = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_PRESENT_SPEED);
                flexNextVel = dynamixel.read2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_PRESENT_SPEED);

                aprbsLog.WriteLine(string.Format("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}", timeStamp, rotCommand, flexCommand, rotCurPos, flexCurPos, rotNextPos, flexNextPos, rotCurVel, flexCurVel, rotNextVel, flexNextVel));

                rotCurPos = rotNextPos;
                flexCurPos = flexNextPos;

                rotCurVel = rotNextVel;
                flexCurVel = flexNextVel;

                rotCommand = command[i];
                flexCommand = command[i];
                timeStamp = DateTime.Now.ToString("HH:mm:ss.fff");

                dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_GOAL_POSITION, rotCommand);
                dynamixel.write2ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_GOAL_POSITION, flexCommand);


                stopWatch1.Stop();
                milliSec1 = stopWatch1.ElapsedMilliseconds;
                if (milliSec1 < 5)
                {
                    Thread.Sleep((int)(5 - milliSec1));
                }

                stopWatch1.Reset();
                stopWatch1.Start();
            }


            // Disable Dynamixel Torque
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 3, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
            dynamixel.write1ByteTxRx(port_num, PROTOCOL_VERSION, 4, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE);
            checkResult(port_num);

            aprbsLog.Dispose();

            // Close port
            dynamixel.closePort(port_num);

            Console.WriteLine("Done");
            Console.ReadKey();

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