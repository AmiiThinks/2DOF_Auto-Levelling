﻿namespace brachIOplexus
{
    partial class mainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(mainForm));
            this.tg = new MathWorks.xPCTarget.FrameWork.xPCTargetPC(this.components);
            this.MenuStrip1 = new System.Windows.Forms.MenuStrip();
            this.FileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.NewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.OpenToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator = new System.Windows.Forms.ToolStripSeparator();
            this.SaveAsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.ToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.ExitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.HelpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.ContentsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mappingGraphicToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.xBoxToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mYOSequentialLeftToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mYOSequentialRightToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.keyboardMultijointToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator5 = new System.Windows.Forms.ToolStripSeparator();
            this.AboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.VoiceCoilCommBox = new System.Windows.Forms.GroupBox();
            this.loadDLMButton = new System.Windows.Forms.Button();
            this.model_name = new System.Windows.Forms.Label();
            this.unloadButton = new System.Windows.Forms.Button();
            this.Label3 = new System.Windows.Forms.Label();
            this.startButton = new System.Windows.Forms.Button();
            this.loadButton = new System.Windows.Forms.Button();
            this.stopButton = new System.Windows.Forms.Button();
            this.disconnectButton = new System.Windows.Forms.Button();
            this.Label9 = new System.Windows.Forms.Label();
            this.ipportTB = new System.Windows.Forms.TextBox();
            this.ipaddressTB = new System.Windows.Forms.TextBox();
            this.Label10 = new System.Windows.Forms.Label();
            this.connectButton = new System.Windows.Forms.Button();
            this.serialPort1 = new System.IO.Ports.SerialPort(this.components);
            this.EMGParamBox = new System.Windows.Forms.GroupBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.pictureBox7 = new System.Windows.Forms.PictureBox();
            this.label78 = new System.Windows.Forms.Label();
            this.DoF4_mode_box = new System.Windows.Forms.ComboBox();
            this.pictureBox8 = new System.Windows.Forms.PictureBox();
            this.label79 = new System.Windows.Forms.Label();
            this.ch8_smax_label = new System.Windows.Forms.Label();
            this.DoF4_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.ch8_smin_label = new System.Windows.Forms.Label();
            this.ch7_smax_label = new System.Windows.Forms.Label();
            this.label83 = new System.Windows.Forms.Label();
            this.ch7_smin_label = new System.Windows.Forms.Label();
            this.ch8_smin_tick = new System.Windows.Forms.Label();
            this.ch8_smax_tick = new System.Windows.Forms.Label();
            this.ch7_smin_tick = new System.Windows.Forms.Label();
            this.ch7_smax_tick = new System.Windows.Forms.Label();
            this.ch8_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch7_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label89 = new System.Windows.Forms.Label();
            this.ch8_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch7_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label90 = new System.Windows.Forms.Label();
            this.label91 = new System.Windows.Forms.Label();
            this.ch8_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label92 = new System.Windows.Forms.Label();
            this.label93 = new System.Windows.Forms.Label();
            this.MAV7_bar = new System.Windows.Forms.ProgressBar();
            this.DoF4_mapping_combobox = new System.Windows.Forms.ComboBox();
            this.label94 = new System.Windows.Forms.Label();
            this.label95 = new System.Windows.Forms.Label();
            this.MAV8_bar = new System.Windows.Forms.ProgressBar();
            this.ch7_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label96 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.pictureBox5 = new System.Windows.Forms.PictureBox();
            this.label51 = new System.Windows.Forms.Label();
            this.DoF3_mode_box = new System.Windows.Forms.ComboBox();
            this.pictureBox6 = new System.Windows.Forms.PictureBox();
            this.label52 = new System.Windows.Forms.Label();
            this.ch6_smax_label = new System.Windows.Forms.Label();
            this.DoF3_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.ch6_smin_label = new System.Windows.Forms.Label();
            this.ch5_smax_label = new System.Windows.Forms.Label();
            this.label57 = new System.Windows.Forms.Label();
            this.ch5_smin_label = new System.Windows.Forms.Label();
            this.ch6_smin_tick = new System.Windows.Forms.Label();
            this.ch6_smax_tick = new System.Windows.Forms.Label();
            this.ch5_smin_tick = new System.Windows.Forms.Label();
            this.ch5_smax_tick = new System.Windows.Forms.Label();
            this.ch6_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch5_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label63 = new System.Windows.Forms.Label();
            this.ch6_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch5_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label64 = new System.Windows.Forms.Label();
            this.label65 = new System.Windows.Forms.Label();
            this.ch6_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label66 = new System.Windows.Forms.Label();
            this.label67 = new System.Windows.Forms.Label();
            this.MAV5_bar = new System.Windows.Forms.ProgressBar();
            this.DoF3_mapping_combobox = new System.Windows.Forms.ComboBox();
            this.label68 = new System.Windows.Forms.Label();
            this.label72 = new System.Windows.Forms.Label();
            this.MAV6_bar = new System.Windows.Forms.ProgressBar();
            this.ch5_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label77 = new System.Windows.Forms.Label();
            this.DoF2box = new System.Windows.Forms.GroupBox();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.label26 = new System.Windows.Forms.Label();
            this.DoF2_mode_box = new System.Windows.Forms.ComboBox();
            this.pictureBox4 = new System.Windows.Forms.PictureBox();
            this.label28 = new System.Windows.Forms.Label();
            this.ch4_smax_label = new System.Windows.Forms.Label();
            this.DoF2_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.ch4_smin_label = new System.Windows.Forms.Label();
            this.ch3_smax_label = new System.Windows.Forms.Label();
            this.label33 = new System.Windows.Forms.Label();
            this.ch3_smin_label = new System.Windows.Forms.Label();
            this.ch4_smin_tick = new System.Windows.Forms.Label();
            this.ch4_smax_tick = new System.Windows.Forms.Label();
            this.ch3_smin_tick = new System.Windows.Forms.Label();
            this.ch3_smax_tick = new System.Windows.Forms.Label();
            this.ch4_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch3_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label43 = new System.Windows.Forms.Label();
            this.ch4_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch3_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label44 = new System.Windows.Forms.Label();
            this.label45 = new System.Windows.Forms.Label();
            this.ch4_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label46 = new System.Windows.Forms.Label();
            this.label47 = new System.Windows.Forms.Label();
            this.MAV3_bar = new System.Windows.Forms.ProgressBar();
            this.DoF2_mapping_combobox = new System.Windows.Forms.ComboBox();
            this.label48 = new System.Windows.Forms.Label();
            this.label49 = new System.Windows.Forms.Label();
            this.MAV4_bar = new System.Windows.Forms.ProgressBar();
            this.ch3_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label50 = new System.Windows.Forms.Label();
            this.DoF1box = new System.Windows.Forms.GroupBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.Label41 = new System.Windows.Forms.Label();
            this.DoF1_mode_box = new System.Windows.Forms.ComboBox();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.Label56 = new System.Windows.Forms.Label();
            this.ch2_smax_label = new System.Windows.Forms.Label();
            this.DoF1_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.ch2_smin_label = new System.Windows.Forms.Label();
            this.ch1_smax_label = new System.Windows.Forms.Label();
            this.label25 = new System.Windows.Forms.Label();
            this.ch1_smin_label = new System.Windows.Forms.Label();
            this.ch2_smin_tick = new System.Windows.Forms.Label();
            this.ch2_smax_tick = new System.Windows.Forms.Label();
            this.ch1_smin_tick = new System.Windows.Forms.Label();
            this.ch1_smax_tick = new System.Windows.Forms.Label();
            this.ch2_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch1_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.Label12 = new System.Windows.Forms.Label();
            this.ch2_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.ch1_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.Label8 = new System.Windows.Forms.Label();
            this.ch2_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.MAV1_bar = new System.Windows.Forms.ProgressBar();
            this.DoF1_mapping_combobox = new System.Windows.Forms.ComboBox();
            this.label14 = new System.Windows.Forms.Label();
            this.label15 = new System.Windows.Forms.Label();
            this.MAV2_bar = new System.Windows.Forms.ProgressBar();
            this.ch1_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label16 = new System.Windows.Forms.Label();
            this.SwitchBox = new System.Windows.Forms.GroupBox();
            this.label35 = new System.Windows.Forms.Label();
            this.switch5_dofmode_box = new System.Windows.Forms.ComboBox();
            this.label34 = new System.Windows.Forms.Label();
            this.switch4_dofmode_box = new System.Windows.Forms.ComboBox();
            this.label32 = new System.Windows.Forms.Label();
            this.switch3_dofmode_box = new System.Windows.Forms.ComboBox();
            this.label30 = new System.Windows.Forms.Label();
            this.switch2_dofmode_box = new System.Windows.Forms.ComboBox();
            this.label24 = new System.Windows.Forms.Label();
            this.cctime_ctrl = new System.Windows.Forms.NumericUpDown();
            this.switch1_dofmode_box = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.text_checkBox = new System.Windows.Forms.CheckBox();
            this.ch9_smax_label = new System.Windows.Forms.Label();
            this.ch9_smin_label = new System.Windows.Forms.Label();
            this.ch9_smin_tick = new System.Windows.Forms.Label();
            this.ch9_smax_tick = new System.Windows.Forms.Label();
            this.ch9_smax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label97 = new System.Windows.Forms.Label();
            this.ch9_smin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label98 = new System.Windows.Forms.Label();
            this.label99 = new System.Windows.Forms.Label();
            this.label100 = new System.Windows.Forms.Label();
            this.MAV9_bar = new System.Windows.Forms.ProgressBar();
            this.ch9_gain_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label101 = new System.Windows.Forms.Label();
            this.led_checkBox = new System.Windows.Forms.CheckBox();
            this.vocal_checkBox = new System.Windows.Forms.CheckBox();
            this.ding_checkBox = new System.Windows.Forms.CheckBox();
            this.label102 = new System.Windows.Forms.Label();
            this.cycle5_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.cycle4_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.cycle3_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.cycle2_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.cycle1_flip_checkBox = new System.Windows.Forms.CheckBox();
            this.label74 = new System.Windows.Forms.Label();
            this.label70 = new System.Windows.Forms.Label();
            this.label31 = new System.Windows.Forms.Label();
            this.switch_mode_combobox = new System.Windows.Forms.ComboBox();
            this.Switch_cycle5_combobox = new System.Windows.Forms.ComboBox();
            this.Switch_cycle4_combobox = new System.Windows.Forms.ComboBox();
            this.Switch_cycle3_combobox = new System.Windows.Forms.ComboBox();
            this.Switch_cycle2_combobox = new System.Windows.Forms.ComboBox();
            this.Switch_cycle1_combobox = new System.Windows.Forms.ComboBox();
            this.cycle_number = new System.Windows.Forms.Label();
            this.label17 = new System.Windows.Forms.Label();
            this.switch_dof_combobox = new System.Windows.Forms.ComboBox();
            this.Label75 = new System.Windows.Forms.Label();
            this.RobotBox = new System.Windows.Forms.GroupBox();
            this.arm_label = new System.Windows.Forms.Label();
            this.RAM_text = new System.Windows.Forms.Label();
            this.label110 = new System.Windows.Forms.Label();
            this.label29 = new System.Windows.Forms.Label();
            this.AX12stopBTN = new System.Windows.Forms.Button();
            this.AX12startBTN = new System.Windows.Forms.Button();
            this.hand_comboBox = new System.Windows.Forms.ComboBox();
            this.label23 = new System.Windows.Forms.Label();
            this.RobotFeedbackBox = new System.Windows.Forms.GroupBox();
            this.Temp5 = new System.Windows.Forms.Label();
            this.Volt5 = new System.Windows.Forms.Label();
            this.Load5 = new System.Windows.Forms.Label();
            this.Vel5 = new System.Windows.Forms.Label();
            this.Pos5 = new System.Windows.Forms.Label();
            this.Temp3 = new System.Windows.Forms.Label();
            this.Volt3 = new System.Windows.Forms.Label();
            this.Load3 = new System.Windows.Forms.Label();
            this.Vel3 = new System.Windows.Forms.Label();
            this.Pos3 = new System.Windows.Forms.Label();
            this.Temp2 = new System.Windows.Forms.Label();
            this.Volt2 = new System.Windows.Forms.Label();
            this.Load2 = new System.Windows.Forms.Label();
            this.Vel2 = new System.Windows.Forms.Label();
            this.Pos2 = new System.Windows.Forms.Label();
            this.Temp1 = new System.Windows.Forms.Label();
            this.Volt1 = new System.Windows.Forms.Label();
            this.Load1 = new System.Windows.Forms.Label();
            this.Vel1 = new System.Windows.Forms.Label();
            this.Pos1 = new System.Windows.Forms.Label();
            this.label109 = new System.Windows.Forms.Label();
            this.label108 = new System.Windows.Forms.Label();
            this.label107 = new System.Windows.Forms.Label();
            this.Temp4 = new System.Windows.Forms.Label();
            this.label106 = new System.Windows.Forms.Label();
            this.label200 = new System.Windows.Forms.Label();
            this.Volt4 = new System.Windows.Forms.Label();
            this.Load4 = new System.Windows.Forms.Label();
            this.Vel4 = new System.Windows.Forms.Label();
            this.Pos4 = new System.Windows.Forms.Label();
            this.RobotParamBox = new System.Windows.Forms.GroupBox();
            this.hand_wmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.hand_wmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.hand_pmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.hand_pmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label7 = new System.Windows.Forms.Label();
            this.wristRot_wmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.wristRot_wmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.Label18 = new System.Windows.Forms.Label();
            this.wristRot_pmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.wristRot_pmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.elbow_wmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.elbow_wmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.elbow_pmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.elbow_pmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.Label20 = new System.Windows.Forms.Label();
            this.shoulder_wmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.shoulder_wmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.shoulder_pmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.shoulder_pmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.Label21 = new System.Windows.Forms.Label();
            this.Label19 = new System.Windows.Forms.Label();
            this.wristFlex_wmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.wristFlex_wmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.wristFlex_pmin_ctrl = new System.Windows.Forms.NumericUpDown();
            this.wristFlex_pmax_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.label22 = new System.Windows.Forms.Label();
            this.SimBox = new System.Windows.Forms.GroupBox();
            this.SIMdcBTN = new System.Windows.Forms.Button();
            this.SIMconnectBTN = new System.Windows.Forms.Button();
            this.openSim = new System.Windows.Forms.Button();
            this.sim_flag = new System.Windows.Forms.CheckBox();
            this.LEDbox = new System.Windows.Forms.GroupBox();
            this.label36 = new System.Windows.Forms.Label();
            this.LEDdisconnect = new System.Windows.Forms.Button();
            this.comboBox1 = new System.Windows.Forms.ComboBox();
            this.LEDconnect = new System.Windows.Forms.Button();
            this.cmbSerialPorts = new System.Windows.Forms.ComboBox();
            this.Timer1 = new System.Windows.Forms.Timer(this.components);
            this.Timer3 = new System.Windows.Forms.Timer(this.components);
            this.SaveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.OpenFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.HelpProvider1 = new System.Windows.Forms.HelpProvider();
            this.Timer2 = new System.Windows.Forms.Timer(this.components);
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.button4 = new System.Windows.Forms.Button();
            this.button5 = new System.Windows.Forms.Button();
            this.button6 = new System.Windows.Forms.Button();
            this.MLBox = new System.Windows.Forms.GroupBox();
            this.home_BTN = new System.Windows.Forms.Button();
            this.torque_off = new System.Windows.Forms.Button();
            this.torque_on = new System.Windows.Forms.Button();
            this.MLdisable = new System.Windows.Forms.Button();
            this.MLenable = new System.Windows.Forms.Button();
            this.ML_stop = new System.Windows.Forms.Button();
            this.ML_start = new System.Windows.Forms.Button();
            this.hand_w = new System.Windows.Forms.NumericUpDown();
            this.hand_p = new System.Windows.Forms.NumericUpDown();
            this.label40 = new System.Windows.Forms.Label();
            this.wristRot_w = new System.Windows.Forms.NumericUpDown();
            this.label42 = new System.Windows.Forms.Label();
            this.wristRot_p = new System.Windows.Forms.NumericUpDown();
            this.elbow_w = new System.Windows.Forms.NumericUpDown();
            this.elbow_p = new System.Windows.Forms.NumericUpDown();
            this.label53 = new System.Windows.Forms.Label();
            this.shoulder_w = new System.Windows.Forms.NumericUpDown();
            this.shoulder_p = new System.Windows.Forms.NumericUpDown();
            this.label54 = new System.Windows.Forms.Label();
            this.label55 = new System.Windows.Forms.Label();
            this.wristFlex_w = new System.Windows.Forms.NumericUpDown();
            this.wristFlex_p = new System.Windows.Forms.NumericUpDown();
            this.label58 = new System.Windows.Forms.Label();
            this.label60 = new System.Windows.Forms.Label();
            this.checkGuide = new System.Windows.Forms.CheckBox();
            this.labelStickRightY = new System.Windows.Forms.Label();
            this.labelStickRightX = new System.Windows.Forms.Label();
            this.labelStickLeftY = new System.Windows.Forms.Label();
            this.labelStickLeftX = new System.Windows.Forms.Label();
            this.labelTriggerRight = new System.Windows.Forms.Label();
            this.labelTriggerLeft = new System.Windows.Forms.Label();
            this.checkDPadLeft = new System.Windows.Forms.CheckBox();
            this.checkDPadDown = new System.Windows.Forms.CheckBox();
            this.checkDPadRight = new System.Windows.Forms.CheckBox();
            this.checkDPadUp = new System.Windows.Forms.CheckBox();
            this.checkStickLeft = new System.Windows.Forms.CheckBox();
            this.checkStickRight = new System.Windows.Forms.CheckBox();
            this.checkBack = new System.Windows.Forms.CheckBox();
            this.checkStart = new System.Windows.Forms.CheckBox();
            this.checkA = new System.Windows.Forms.CheckBox();
            this.checkB = new System.Windows.Forms.CheckBox();
            this.checkX = new System.Windows.Forms.CheckBox();
            this.checkY = new System.Windows.Forms.CheckBox();
            this.checkShoulderRight = new System.Windows.Forms.CheckBox();
            this.label59 = new System.Windows.Forms.Label();
            this.label61 = new System.Windows.Forms.Label();
            this.label62 = new System.Windows.Forms.Label();
            this.label69 = new System.Windows.Forms.Label();
            this.label71 = new System.Windows.Forms.Label();
            this.label73 = new System.Windows.Forms.Label();
            this.label76 = new System.Windows.Forms.Label();
            this.label80 = new System.Windows.Forms.Label();
            this.label81 = new System.Windows.Forms.Label();
            this.label82 = new System.Windows.Forms.Label();
            this.label84 = new System.Windows.Forms.Label();
            this.label85 = new System.Windows.Forms.Label();
            this.label86 = new System.Windows.Forms.Label();
            this.label87 = new System.Windows.Forms.Label();
            this.label88 = new System.Windows.Forms.Label();
            this.checkShoulderLeft = new System.Windows.Forms.CheckBox();
            this.label105 = new System.Windows.Forms.Label();
            this.label111 = new System.Windows.Forms.Label();
            this.label112 = new System.Windows.Forms.Label();
            this.label113 = new System.Windows.Forms.Label();
            this.label114 = new System.Windows.Forms.Label();
            this.label115 = new System.Windows.Forms.Label();
            this.pollingWorker = new System.ComponentModel.BackgroundWorker();
            this.dynaConnect = new System.Windows.Forms.Button();
            this.dynaDisconnect = new System.Windows.Forms.Button();
            this.TorqueOn = new System.Windows.Forms.Button();
            this.TorqueOff = new System.Windows.Forms.Button();
            this.LEDon = new System.Windows.Forms.Button();
            this.LEDoff = new System.Windows.Forms.Button();
            this.moveCW = new System.Windows.Forms.Button();
            this.moveCCW = new System.Windows.Forms.Button();
            this.label116 = new System.Windows.Forms.Label();
            this.label117 = new System.Windows.Forms.Label();
            this.delay = new System.Windows.Forms.Label();
            this.label118 = new System.Windows.Forms.Label();
            this.dynaCommResult = new System.Windows.Forms.Label();
            this.label120 = new System.Windows.Forms.Label();
            this.dynaError = new System.Windows.Forms.Label();
            this.readFeedback = new System.Windows.Forms.Button();
            this.delay_max = new System.Windows.Forms.Label();
            this.label121 = new System.Windows.Forms.Label();
            this.label119 = new System.Windows.Forms.Label();
            this.dynaStatus = new System.Windows.Forms.Label();
            this.cmbSerialRefresh = new System.Windows.Forms.Button();
            this.BentoGroupBox = new System.Windows.Forms.GroupBox();
            this.LogPID_Enabled = new System.Windows.Forms.CheckBox();
            this.label160 = new System.Windows.Forms.Label();
            this.BentoRun = new System.Windows.Forms.Button();
            this.BentoSuspend = new System.Windows.Forms.Button();
            this.xBoxGroupBox = new System.Windows.Forms.GroupBox();
            this.XboxDisconnect = new System.Windows.Forms.Button();
            this.XboxConnect = new System.Windows.Forms.Button();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.groupBox18 = new System.Windows.Forms.GroupBox();
            this.ArduinoInputCOM = new System.Windows.Forms.ComboBox();
            this.label204 = new System.Windows.Forms.Label();
            this.ArduinoInputClearAll = new System.Windows.Forms.Button();
            this.ArduinoInputConnect = new System.Windows.Forms.Button();
            this.ArduinoInputSelectAll = new System.Windows.Forms.Button();
            this.ArduinoInputDisconnect = new System.Windows.Forms.Button();
            this.ArduinoInputList = new System.Windows.Forms.CheckedListBox();
            this.groupBox17 = new System.Windows.Forms.GroupBox();
            this.SLRTclearAll = new System.Windows.Forms.Button();
            this.SLRTconnect = new System.Windows.Forms.Button();
            this.SLRTselectAll = new System.Windows.Forms.Button();
            this.SLRTdisconnect = new System.Windows.Forms.Button();
            this.SLRTlist = new System.Windows.Forms.CheckedListBox();
            this.groupBox15 = new System.Windows.Forms.GroupBox();
            this.biopatrecMode = new System.Windows.Forms.ComboBox();
            this.label202 = new System.Windows.Forms.Label();
            this.biopatrecIPport = new System.Windows.Forms.TextBox();
            this.label186 = new System.Windows.Forms.Label();
            this.label188 = new System.Windows.Forms.Label();
            this.biopatrecClearAll = new System.Windows.Forms.Button();
            this.biopatrecConnect = new System.Windows.Forms.Button();
            this.biopatrecIPaddr = new System.Windows.Forms.TextBox();
            this.biopatrecSelectAll = new System.Windows.Forms.Button();
            this.biopatrecDisconnect = new System.Windows.Forms.Button();
            this.biopatrecList = new System.Windows.Forms.CheckedListBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.pictureBox10 = new System.Windows.Forms.PictureBox();
            this.MYOclearAll = new System.Windows.Forms.Button();
            this.MYOconnect = new System.Windows.Forms.Button();
            this.MYOselectAll = new System.Windows.Forms.Button();
            this.MYOdisconnect = new System.Windows.Forms.Button();
            this.MYOlist = new System.Windows.Forms.CheckedListBox();
            this.groupBox8 = new System.Windows.Forms.GroupBox();
            this.pictureBox11 = new System.Windows.Forms.PictureBox();
            this.KBclearAll = new System.Windows.Forms.Button();
            this.KBlist = new System.Windows.Forms.CheckedListBox();
            this.KBselectAll = new System.Windows.Forms.Button();
            this.KBcheckRamp = new System.Windows.Forms.CheckBox();
            this.KBlabelRamp = new System.Windows.Forms.Label();
            this.KBconnect = new System.Windows.Forms.Button();
            this.KBdisconnect = new System.Windows.Forms.Button();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.pictureBox9 = new System.Windows.Forms.PictureBox();
            this.XBoxClearAll = new System.Windows.Forms.Button();
            this.XBoxSelectAll = new System.Windows.Forms.Button();
            this.XBoxList = new System.Windows.Forms.CheckedListBox();
            this.label203 = new System.Windows.Forms.Label();
            this.biopatrecDelay = new System.Windows.Forms.Label();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.pictureBox12 = new System.Windows.Forms.PictureBox();
            this.BentoClearAll = new System.Windows.Forms.Button();
            this.BentoSelectAll = new System.Windows.Forms.Button();
            this.BentoList = new System.Windows.Forms.CheckedListBox();
            this.button1 = new System.Windows.Forms.Button();
            this.comboBox2 = new System.Windows.Forms.ComboBox();
            this.checkedListFruit = new System.Windows.Forms.CheckedListBox();
            this.MYOgroupBox = new System.Windows.Forms.GroupBox();
            this.myo_ch1 = new System.Windows.Forms.Label();
            this.myo_ch2 = new System.Windows.Forms.Label();
            this.label134 = new System.Windows.Forms.Label();
            this.label136 = new System.Windows.Forms.Label();
            this.myo_ch3 = new System.Windows.Forms.Label();
            this.myo_ch4 = new System.Windows.Forms.Label();
            this.myo_ch5 = new System.Windows.Forms.Label();
            this.myo_ch6 = new System.Windows.Forms.Label();
            this.myo_ch7 = new System.Windows.Forms.Label();
            this.label128 = new System.Windows.Forms.Label();
            this.myo_ch8 = new System.Windows.Forms.Label();
            this.label130 = new System.Windows.Forms.Label();
            this.label131 = new System.Windows.Forms.Label();
            this.label133 = new System.Windows.Forms.Label();
            this.label135 = new System.Windows.Forms.Label();
            this.label137 = new System.Windows.Forms.Label();
            this.KBgroupBox = new System.Windows.Forms.GroupBox();
            this.KBrampS = new System.Windows.Forms.Label();
            this.KBrampD = new System.Windows.Forms.Label();
            this.KBrampW = new System.Windows.Forms.Label();
            this.KBcheckRightAlt = new System.Windows.Forms.CheckBox();
            this.KBrampA = new System.Windows.Forms.Label();
            this.KBcheckSpace = new System.Windows.Forms.CheckBox();
            this.KBcheckLeftAlt = new System.Windows.Forms.CheckBox();
            this.label142 = new System.Windows.Forms.Label();
            this.label143 = new System.Windows.Forms.Label();
            this.label144 = new System.Windows.Forms.Label();
            this.KBcheckRight = new System.Windows.Forms.CheckBox();
            this.KBcheckDown = new System.Windows.Forms.CheckBox();
            this.KBcheckLeft = new System.Windows.Forms.CheckBox();
            this.KBcheckUp = new System.Windows.Forms.CheckBox();
            this.label138 = new System.Windows.Forms.Label();
            this.label139 = new System.Windows.Forms.Label();
            this.label140 = new System.Windows.Forms.Label();
            this.label141 = new System.Windows.Forms.Label();
            this.KBcheckSemiColon = new System.Windows.Forms.CheckBox();
            this.KBcheckL = new System.Windows.Forms.CheckBox();
            this.KBcheckK = new System.Windows.Forms.CheckBox();
            this.KBcheckO = new System.Windows.Forms.CheckBox();
            this.label126 = new System.Windows.Forms.Label();
            this.label127 = new System.Windows.Forms.Label();
            this.label129 = new System.Windows.Forms.Label();
            this.label132 = new System.Windows.Forms.Label();
            this.KBcheckD = new System.Windows.Forms.CheckBox();
            this.KBcheckS = new System.Windows.Forms.CheckBox();
            this.KBcheckA = new System.Windows.Forms.CheckBox();
            this.KBcheckW = new System.Windows.Forms.CheckBox();
            this.label122 = new System.Windows.Forms.Label();
            this.label123 = new System.Windows.Forms.Label();
            this.label124 = new System.Windows.Forms.Label();
            this.label125 = new System.Windows.Forms.Label();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabIO = new System.Windows.Forms.TabPage();
            this.demoShutdownButton = new System.Windows.Forms.Button();
            this.demoMYObutton = new System.Windows.Forms.Button();
            this.demoXBoxButton = new System.Windows.Forms.Button();
            this.InputComboBox = new System.Windows.Forms.ComboBox();
            this.OutputComboBox = new System.Windows.Forms.ComboBox();
            this.label166 = new System.Windows.Forms.Label();
            this.labelType = new System.Windows.Forms.Label();
            this.checkedListDairy = new System.Windows.Forms.CheckedListBox();
            this.button14 = new System.Windows.Forms.Button();
            this.labelID = new System.Windows.Forms.Label();
            this.labelText = new System.Windows.Forms.Label();
            this.groupBox9 = new System.Windows.Forms.GroupBox();
            this.tabMapping = new System.Windows.Forms.TabPage();
            this.LoggingGroupBox = new System.Windows.Forms.GroupBox();
            this.label234 = new System.Windows.Forms.Label();
            this.label233 = new System.Windows.Forms.Label();
            this.label232 = new System.Windows.Forms.Label();
            this.label231 = new System.Windows.Forms.Label();
            this.label230 = new System.Windows.Forms.Label();
            this.intervention = new System.Windows.Forms.TextBox();
            this.label229 = new System.Windows.Forms.Label();
            this.task_type = new System.Windows.Forms.TextBox();
            this.label227 = new System.Windows.Forms.Label();
            this.ppt_no = new System.Windows.Forms.TextBox();
            this.label_ppt_no = new System.Windows.Forms.Label();
            this.StartLogging = new System.Windows.Forms.Button();
            this.label228 = new System.Windows.Forms.Label();
            this.StopLogging = new System.Windows.Forms.Button();
            this.log_number = new System.Windows.Forms.NumericUpDown();
            this.groupBox16 = new System.Windows.Forms.GroupBox();
            this.switchSmaxLabel2 = new System.Windows.Forms.Label();
            this.switchSminLabel2 = new System.Windows.Forms.Label();
            this.switchSminTick2 = new System.Windows.Forms.Label();
            this.switchState_label = new System.Windows.Forms.Label();
            this.switchSmaxTick2 = new System.Windows.Forms.Label();
            this.label213 = new System.Windows.Forms.Label();
            this.switchSmaxCtrl2 = new System.Windows.Forms.NumericUpDown();
            this.flag2_label = new System.Windows.Forms.Label();
            this.switchSminCtrl2 = new System.Windows.Forms.NumericUpDown();
            this.label211 = new System.Windows.Forms.Label();
            this.switchSignalBar2 = new System.Windows.Forms.ProgressBar();
            this.flag1_label = new System.Windows.Forms.Label();
            this.switchGainCtrl2 = new System.Windows.Forms.NumericUpDown();
            this.label209 = new System.Windows.Forms.Label();
            this.switchTimeCtrl2 = new System.Windows.Forms.NumericUpDown();
            this.timer1_label = new System.Windows.Forms.Label();
            this.groupBox11 = new System.Windows.Forms.GroupBox();
            this.groupBox14 = new System.Windows.Forms.GroupBox();
            this.textBox = new System.Windows.Forms.CheckBox();
            this.groupBox13 = new System.Windows.Forms.GroupBox();
            this.XboxBuzzBox = new System.Windows.Forms.CheckBox();
            this.myoBuzzBox = new System.Windows.Forms.CheckBox();
            this.groupBox12 = new System.Windows.Forms.GroupBox();
            this.dingBox = new System.Windows.Forms.CheckBox();
            this.vocalBox = new System.Windows.Forms.CheckBox();
            this.label205 = new System.Windows.Forms.Label();
            this.ID2_state = new System.Windows.Forms.Label();
            this.groupBox10 = new System.Windows.Forms.GroupBox();
            this.label37 = new System.Windows.Forms.Label();
            this.switch1Flip = new System.Windows.Forms.CheckBox();
            this.switch1OutputBox = new System.Windows.Forms.ComboBox();
            this.label238 = new System.Windows.Forms.Label();
            this.switch2Flip = new System.Windows.Forms.CheckBox();
            this.switch5MappingBox = new System.Windows.Forms.ComboBox();
            this.switch4MappingBox = new System.Windows.Forms.ComboBox();
            this.switch2OutputBox = new System.Windows.Forms.ComboBox();
            this.switch3Flip = new System.Windows.Forms.CheckBox();
            this.label240 = new System.Windows.Forms.Label();
            this.label253 = new System.Windows.Forms.Label();
            this.label239 = new System.Windows.Forms.Label();
            this.label38 = new System.Windows.Forms.Label();
            this.switch5Flip = new System.Windows.Forms.CheckBox();
            this.label241 = new System.Windows.Forms.Label();
            this.label237 = new System.Windows.Forms.Label();
            this.switch1MappingBox = new System.Windows.Forms.ComboBox();
            this.switch3OutputBox = new System.Windows.Forms.ComboBox();
            this.switch4Flip = new System.Windows.Forms.CheckBox();
            this.switch3MappingBox = new System.Windows.Forms.ComboBox();
            this.switch5OutputBox = new System.Windows.Forms.ComboBox();
            this.switch2MappingBox = new System.Windows.Forms.ComboBox();
            this.switch4OutputBox = new System.Windows.Forms.ComboBox();
            this.label148 = new System.Windows.Forms.Label();
            this.label103 = new System.Windows.Forms.Label();
            this.label104 = new System.Windows.Forms.Label();
            this.label145 = new System.Windows.Forms.Label();
            this.label147 = new System.Windows.Forms.Label();
            this.switchSmaxLabel1 = new System.Windows.Forms.Label();
            this.switchSminLabel1 = new System.Windows.Forms.Label();
            this.switchSminTick1 = new System.Windows.Forms.Label();
            this.switchSmaxTick1 = new System.Windows.Forms.Label();
            this.switchSmaxCtrl1 = new System.Windows.Forms.NumericUpDown();
            this.switchSminCtrl1 = new System.Windows.Forms.NumericUpDown();
            this.switchSignalBar1 = new System.Windows.Forms.ProgressBar();
            this.switchGainCtrl1 = new System.Windows.Forms.NumericUpDown();
            this.switchInputBox = new System.Windows.Forms.ComboBox();
            this.label39 = new System.Windows.Forms.Label();
            this.label27 = new System.Windows.Forms.Label();
            this.switchTimeCtrl1 = new System.Windows.Forms.NumericUpDown();
            this.label242 = new System.Windows.Forms.Label();
            this.switchModeBox = new System.Windows.Forms.ComboBox();
            this.switchLabel = new System.Windows.Forms.Label();
            this.label257 = new System.Windows.Forms.Label();
            this.switchDoFbox = new System.Windows.Forms.ComboBox();
            this.label258 = new System.Windows.Forms.Label();
            this.label162 = new System.Windows.Forms.Label();
            this.label150 = new System.Windows.Forms.Label();
            this.label146 = new System.Windows.Forms.Label();
            this.label163 = new System.Windows.Forms.Label();
            this.label158 = new System.Windows.Forms.Label();
            this.label157 = new System.Windows.Forms.Label();
            this.label156 = new System.Windows.Forms.Label();
            this.doF6 = new brachIOplexus.DoF();
            this.doF5 = new brachIOplexus.DoF();
            this.doF4 = new brachIOplexus.DoF();
            this.doF3 = new brachIOplexus.DoF();
            this.doF2 = new brachIOplexus.DoF();
            this.doF1 = new brachIOplexus.DoF();
            this.tabBento = new System.Windows.Forms.TabPage();
            this.AutoLevellingBox = new System.Windows.Forms.GroupBox();
            this.NN_PID_Enabled = new System.Windows.Forms.CheckBox();
            this.FlexionPIDBox = new System.Windows.Forms.GroupBox();
            this.CurrentFlexion = new System.Windows.Forms.Label();
            this.label224 = new System.Windows.Forms.Label();
            this.label223 = new System.Windows.Forms.Label();
            this.SetpointFlexion = new System.Windows.Forms.Label();
            this.Kd_theta_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label210 = new System.Windows.Forms.Label();
            this.Ki_theta_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label206 = new System.Windows.Forms.Label();
            this.Kp_theta_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label212 = new System.Windows.Forms.Label();
            this.RotationPIDBox = new System.Windows.Forms.GroupBox();
            this.CurrentRotation = new System.Windows.Forms.Label();
            this.label222 = new System.Windows.Forms.Label();
            this.label221 = new System.Windows.Forms.Label();
            this.Kd_phi_ctrl = new System.Windows.Forms.NumericUpDown();
            this.SetpointRotation = new System.Windows.Forms.Label();
            this.label215 = new System.Windows.Forms.Label();
            this.Ki_phi_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label225 = new System.Windows.Forms.Label();
            this.Kp_phi_ctrl = new System.Windows.Forms.NumericUpDown();
            this.label226 = new System.Windows.Forms.Label();
            this.AL_Enabled = new System.Windows.Forms.CheckBox();
            this.groupBox19 = new System.Windows.Forms.GroupBox();
            this.BentoProfileOpen = new System.Windows.Forms.Button();
            this.BentoProfileBox = new System.Windows.Forms.ComboBox();
            this.BentoProfileSave = new System.Windows.Forms.Button();
            this.BentoEnvLimitsBox = new System.Windows.Forms.GroupBox();
            this.label159 = new System.Windows.Forms.Label();
            this.label155 = new System.Windows.Forms.Label();
            this.numericUpDown3 = new System.Windows.Forms.NumericUpDown();
            this.label153 = new System.Windows.Forms.Label();
            this.environCheck = new System.Windows.Forms.CheckBox();
            this.numericUpDown2 = new System.Windows.Forms.NumericUpDown();
            this.label154 = new System.Windows.Forms.Label();
            this.BentoAdaptGripBox = new System.Windows.Forms.GroupBox();
            this.BentoAdaptGripCheck = new System.Windows.Forms.CheckBox();
            this.label152 = new System.Windows.Forms.Label();
            this.BentoAdaptGripCtrl = new System.Windows.Forms.NumericUpDown();
            this.label151 = new System.Windows.Forms.Label();
            this.tabXPC = new System.Windows.Forms.TabPage();
            this.tabViz = new System.Windows.Forms.TabPage();
            this.ArduinoInputGroupBox = new System.Windows.Forms.GroupBox();
            this.arduino_A0 = new System.Windows.Forms.Label();
            this.arduino_A1 = new System.Windows.Forms.Label();
            this.label207 = new System.Windows.Forms.Label();
            this.label208 = new System.Windows.Forms.Label();
            this.arduino_A2 = new System.Windows.Forms.Label();
            this.arduino_A3 = new System.Windows.Forms.Label();
            this.arduino_A4 = new System.Windows.Forms.Label();
            this.arduino_A5 = new System.Windows.Forms.Label();
            this.arduino_A6 = new System.Windows.Forms.Label();
            this.label214 = new System.Windows.Forms.Label();
            this.arduino_A7 = new System.Windows.Forms.Label();
            this.label216 = new System.Windows.Forms.Label();
            this.label217 = new System.Windows.Forms.Label();
            this.label218 = new System.Windows.Forms.Label();
            this.label219 = new System.Windows.Forms.Label();
            this.label220 = new System.Windows.Forms.Label();
            this.biopatrecGroupBox = new System.Windows.Forms.GroupBox();
            this.label184 = new System.Windows.Forms.Label();
            this.label182 = new System.Windows.Forms.Label();
            this.BPRclass12 = new System.Windows.Forms.CheckBox();
            this.label165 = new System.Windows.Forms.Label();
            this.BPRclass24 = new System.Windows.Forms.CheckBox();
            this.label169 = new System.Windows.Forms.Label();
            this.BPRclass23 = new System.Windows.Forms.CheckBox();
            this.BPRclass17 = new System.Windows.Forms.CheckBox();
            this.BPRclass18 = new System.Windows.Forms.CheckBox();
            this.BPRclass21 = new System.Windows.Forms.CheckBox();
            this.BPRclass20 = new System.Windows.Forms.CheckBox();
            this.BPRclass19 = new System.Windows.Forms.CheckBox();
            this.BPRclass22 = new System.Windows.Forms.CheckBox();
            this.label170 = new System.Windows.Forms.Label();
            this.label171 = new System.Windows.Forms.Label();
            this.label172 = new System.Windows.Forms.Label();
            this.label173 = new System.Windows.Forms.Label();
            this.label175 = new System.Windows.Forms.Label();
            this.label181 = new System.Windows.Forms.Label();
            this.label201 = new System.Windows.Forms.Label();
            this.BPRclass11 = new System.Windows.Forms.CheckBox();
            this.label161 = new System.Windows.Forms.Label();
            this.BPRclass10 = new System.Windows.Forms.CheckBox();
            this.BPRclass3 = new System.Windows.Forms.CheckBox();
            this.BPRclass2 = new System.Windows.Forms.CheckBox();
            this.BPRclass1 = new System.Windows.Forms.CheckBox();
            this.BPRclass0 = new System.Windows.Forms.CheckBox();
            this.BPRclass4 = new System.Windows.Forms.CheckBox();
            this.BPRclass5 = new System.Windows.Forms.CheckBox();
            this.BPRclass8 = new System.Windows.Forms.CheckBox();
            this.BPRclass7 = new System.Windows.Forms.CheckBox();
            this.BPRclass13 = new System.Windows.Forms.CheckBox();
            this.BPRclass14 = new System.Windows.Forms.CheckBox();
            this.BPRclass15 = new System.Windows.Forms.CheckBox();
            this.BPRclass16 = new System.Windows.Forms.CheckBox();
            this.BPRclass6 = new System.Windows.Forms.CheckBox();
            this.label183 = new System.Windows.Forms.Label();
            this.label185 = new System.Windows.Forms.Label();
            this.label187 = new System.Windows.Forms.Label();
            this.label189 = new System.Windows.Forms.Label();
            this.BPRclass9 = new System.Windows.Forms.CheckBox();
            this.label190 = new System.Windows.Forms.Label();
            this.label191 = new System.Windows.Forms.Label();
            this.label192 = new System.Windows.Forms.Label();
            this.label193 = new System.Windows.Forms.Label();
            this.label194 = new System.Windows.Forms.Label();
            this.label195 = new System.Windows.Forms.Label();
            this.label196 = new System.Windows.Forms.Label();
            this.label197 = new System.Windows.Forms.Label();
            this.label198 = new System.Windows.Forms.Label();
            this.label199 = new System.Windows.Forms.Label();
            this.SLRTgroupBox = new System.Windows.Forms.GroupBox();
            this.slrt_ch1 = new System.Windows.Forms.Label();
            this.slrt_ch2 = new System.Windows.Forms.Label();
            this.label167 = new System.Windows.Forms.Label();
            this.label168 = new System.Windows.Forms.Label();
            this.slrt_ch3 = new System.Windows.Forms.Label();
            this.slrt_ch4 = new System.Windows.Forms.Label();
            this.slrt_ch5 = new System.Windows.Forms.Label();
            this.slrt_ch6 = new System.Windows.Forms.Label();
            this.slrt_ch7 = new System.Windows.Forms.Label();
            this.label174 = new System.Windows.Forms.Label();
            this.slrt_ch8 = new System.Windows.Forms.Label();
            this.label176 = new System.Windows.Forms.Label();
            this.label177 = new System.Windows.Forms.Label();
            this.label178 = new System.Windows.Forms.Label();
            this.label179 = new System.Windows.Forms.Label();
            this.label180 = new System.Windows.Forms.Label();
            this.statusPanel1 = new System.Windows.Forms.Panel();
            this.MYOstatus = new System.Windows.Forms.Label();
            this.BentoErrorText = new System.Windows.Forms.Label();
            this.label164 = new System.Windows.Forms.Label();
            this.BentoErrorColor = new System.Windows.Forms.Label();
            this.BentoRunStatus = new System.Windows.Forms.Button();
            this.BentoStatus = new System.Windows.Forms.Label();
            this.label149 = new System.Windows.Forms.Label();
            this.serialArduinoInput = new System.IO.Ports.SerialPort(this.components);
            this.MenuStrip1.SuspendLayout();
            this.VoiceCoilCommBox.SuspendLayout();
            this.EMGParamBox.SuspendLayout();
            this.groupBox3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox8)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_gain_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_gain_ctrl)).BeginInit();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_gain_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_gain_ctrl)).BeginInit();
            this.DoF2box.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_gain_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_gain_ctrl)).BeginInit();
            this.DoF1box.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_gain_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_gain_ctrl)).BeginInit();
            this.SwitchBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.cctime_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_smax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_smin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_gain_ctrl)).BeginInit();
            this.RobotBox.SuspendLayout();
            this.RobotFeedbackBox.SuspendLayout();
            this.RobotParamBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.hand_wmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_wmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_pmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_pmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_wmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_wmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_pmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_pmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_wmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_wmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_pmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_pmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_wmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_wmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_pmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_pmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_wmax_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_wmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_pmin_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_pmax_ctrl)).BeginInit();
            this.SimBox.SuspendLayout();
            this.LEDbox.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.MLBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.hand_w)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_p)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_w)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_p)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_w)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_p)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_w)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_p)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_w)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_p)).BeginInit();
            this.BentoGroupBox.SuspendLayout();
            this.xBoxGroupBox.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox18.SuspendLayout();
            this.groupBox17.SuspendLayout();
            this.groupBox15.SuspendLayout();
            this.groupBox7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox10)).BeginInit();
            this.groupBox8.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox11)).BeginInit();
            this.groupBox5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox9)).BeginInit();
            this.groupBox6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox12)).BeginInit();
            this.MYOgroupBox.SuspendLayout();
            this.KBgroupBox.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabIO.SuspendLayout();
            this.groupBox9.SuspendLayout();
            this.tabMapping.SuspendLayout();
            this.LoggingGroupBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.log_number)).BeginInit();
            this.groupBox16.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.switchSmaxCtrl2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchSminCtrl2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchGainCtrl2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchTimeCtrl2)).BeginInit();
            this.groupBox11.SuspendLayout();
            this.groupBox14.SuspendLayout();
            this.groupBox13.SuspendLayout();
            this.groupBox12.SuspendLayout();
            this.groupBox10.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.switchSmaxCtrl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchSminCtrl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchGainCtrl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchTimeCtrl1)).BeginInit();
            this.tabBento.SuspendLayout();
            this.AutoLevellingBox.SuspendLayout();
            this.FlexionPIDBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Kd_theta_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Ki_theta_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Kp_theta_ctrl)).BeginInit();
            this.RotationPIDBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Kd_phi_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Ki_phi_ctrl)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Kp_phi_ctrl)).BeginInit();
            this.groupBox19.SuspendLayout();
            this.BentoEnvLimitsBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).BeginInit();
            this.BentoAdaptGripBox.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.BentoAdaptGripCtrl)).BeginInit();
            this.tabXPC.SuspendLayout();
            this.tabViz.SuspendLayout();
            this.ArduinoInputGroupBox.SuspendLayout();
            this.biopatrecGroupBox.SuspendLayout();
            this.SLRTgroupBox.SuspendLayout();
            this.statusPanel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // tg
            // 
            //this.tg.ContainerControl = this;
            //this.tg.DLMFileName = null;
            //this.tg.HostTargetComm = MathWorks.xPCTarget.FrameWork.XPCProtocol.TCPIP;
            //this.tg.RS232Baudrate = MathWorks.xPCTarget.FrameWork.XPCRS232BaudRate.BAUD115200;
            //this.tg.RS232HostPort = MathWorks.xPCTarget.FrameWork.XPCRS232CommPort.COM1;
            //this.tg.TargetPCName = null;
            //this.tg.TcpIpTargetAddress = "10.10.10.15";
            //this.tg.TcpIpTargetPort = "22222";
            // 
            // MenuStrip1
            // 
            this.MenuStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.MenuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.FileToolStripMenuItem,
            this.HelpToolStripMenuItem});
            this.MenuStrip1.Location = new System.Drawing.Point(0, 0);
            this.MenuStrip1.Name = "MenuStrip1";
            this.MenuStrip1.Padding = new System.Windows.Forms.Padding(5, 2, 0, 2);
            this.MenuStrip1.Size = new System.Drawing.Size(1576, 28);
            this.MenuStrip1.TabIndex = 54;
            this.MenuStrip1.Text = "MenuStrip1";
            // 
            // FileToolStripMenuItem
            // 
            this.FileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.NewToolStripMenuItem,
            this.OpenToolStripMenuItem,
            this.toolStripSeparator,
            this.SaveAsToolStripMenuItem,
            this.ToolStripMenuItem1,
            this.toolStripSeparator2,
            this.ExitToolStripMenuItem});
            this.FileToolStripMenuItem.Name = "FileToolStripMenuItem";
            this.FileToolStripMenuItem.Size = new System.Drawing.Size(46, 24);
            this.FileToolStripMenuItem.Text = "&File";
            // 
            // NewToolStripMenuItem
            // 
            this.NewToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("NewToolStripMenuItem.Image")));
            this.NewToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.NewToolStripMenuItem.Name = "NewToolStripMenuItem";
            this.NewToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.N)));
            this.NewToolStripMenuItem.Size = new System.Drawing.Size(240, 26);
            this.NewToolStripMenuItem.Text = "&New Profile";
            this.NewToolStripMenuItem.Click += new System.EventHandler(this.NewToolStripMenuItem_Click);
            // 
            // OpenToolStripMenuItem
            // 
            this.OpenToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("OpenToolStripMenuItem.Image")));
            this.OpenToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.OpenToolStripMenuItem.Name = "OpenToolStripMenuItem";
            this.OpenToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.OpenToolStripMenuItem.Size = new System.Drawing.Size(240, 26);
            this.OpenToolStripMenuItem.Text = "&Open Profile";
            this.OpenToolStripMenuItem.Click += new System.EventHandler(this.OpenToolStripMenuItem_Click);
            // 
            // toolStripSeparator
            // 
            this.toolStripSeparator.Name = "toolStripSeparator";
            this.toolStripSeparator.Size = new System.Drawing.Size(237, 6);
            // 
            // SaveAsToolStripMenuItem
            // 
            this.SaveAsToolStripMenuItem.Name = "SaveAsToolStripMenuItem";
            this.SaveAsToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.SaveAsToolStripMenuItem.Size = new System.Drawing.Size(240, 26);
            this.SaveAsToolStripMenuItem.Text = "&Save Profile As";
            this.SaveAsToolStripMenuItem.Click += new System.EventHandler(this.SaveAsToolStripMenuItem_Click);
            // 
            // ToolStripMenuItem1
            // 
            this.ToolStripMenuItem1.Name = "ToolStripMenuItem1";
            this.ToolStripMenuItem1.Size = new System.Drawing.Size(240, 26);
            this.ToolStripMenuItem1.Text = "Set As Default Profile";
            this.ToolStripMenuItem1.Click += new System.EventHandler(this.ToolStripMenuItem1_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(237, 6);
            // 
            // ExitToolStripMenuItem
            // 
            this.ExitToolStripMenuItem.Name = "ExitToolStripMenuItem";
            this.ExitToolStripMenuItem.Size = new System.Drawing.Size(240, 26);
            this.ExitToolStripMenuItem.Text = "E&xit";
            this.ExitToolStripMenuItem.Click += new System.EventHandler(this.ExitToolStripMenuItem_Click);
            // 
            // HelpToolStripMenuItem
            // 
            this.HelpToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.ContentsToolStripMenuItem,
            this.mappingGraphicToolStripMenuItem,
            this.toolStripSeparator5,
            this.AboutToolStripMenuItem});
            this.HelpToolStripMenuItem.Name = "HelpToolStripMenuItem";
            this.HelpToolStripMenuItem.Size = new System.Drawing.Size(55, 24);
            this.HelpToolStripMenuItem.Text = "&Help";
            // 
            // ContentsToolStripMenuItem
            // 
            this.ContentsToolStripMenuItem.Name = "ContentsToolStripMenuItem";
            this.ContentsToolStripMenuItem.Size = new System.Drawing.Size(220, 26);
            this.ContentsToolStripMenuItem.Text = "&User Manual";
            this.ContentsToolStripMenuItem.Click += new System.EventHandler(this.ContentsToolStripMenuItem_Click);
            // 
            // mappingGraphicToolStripMenuItem
            // 
            this.mappingGraphicToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.xBoxToolStripMenuItem,
            this.mYOSequentialLeftToolStripMenuItem,
            this.mYOSequentialRightToolStripMenuItem,
            this.keyboardMultijointToolStripMenuItem});
            this.mappingGraphicToolStripMenuItem.Name = "mappingGraphicToolStripMenuItem";
            this.mappingGraphicToolStripMenuItem.Size = new System.Drawing.Size(220, 26);
            this.mappingGraphicToolStripMenuItem.Text = "Mapping Diagrams";
            // 
            // xBoxToolStripMenuItem
            // 
            this.xBoxToolStripMenuItem.Name = "xBoxToolStripMenuItem";
            this.xBoxToolStripMenuItem.Size = new System.Drawing.Size(236, 26);
            this.xBoxToolStripMenuItem.Text = "XBox Multijoint";
            this.xBoxToolStripMenuItem.Click += new System.EventHandler(this.xBoxToolStripMenuItem_Click);
            // 
            // mYOSequentialLeftToolStripMenuItem
            // 
            this.mYOSequentialLeftToolStripMenuItem.Name = "mYOSequentialLeftToolStripMenuItem";
            this.mYOSequentialLeftToolStripMenuItem.Size = new System.Drawing.Size(236, 26);
            this.mYOSequentialLeftToolStripMenuItem.Text = "MYO Sequential Left";
            this.mYOSequentialLeftToolStripMenuItem.Click += new System.EventHandler(this.mYOSequentialLeftToolStripMenuItem_Click);
            // 
            // mYOSequentialRightToolStripMenuItem
            // 
            this.mYOSequentialRightToolStripMenuItem.Name = "mYOSequentialRightToolStripMenuItem";
            this.mYOSequentialRightToolStripMenuItem.Size = new System.Drawing.Size(236, 26);
            this.mYOSequentialRightToolStripMenuItem.Text = "MYO Sequential Right";
            this.mYOSequentialRightToolStripMenuItem.Click += new System.EventHandler(this.mYOSequentialRightToolStripMenuItem_Click);
            // 
            // keyboardMultijointToolStripMenuItem
            // 
            this.keyboardMultijointToolStripMenuItem.Name = "keyboardMultijointToolStripMenuItem";
            this.keyboardMultijointToolStripMenuItem.Size = new System.Drawing.Size(236, 26);
            this.keyboardMultijointToolStripMenuItem.Text = "Keyboard Multijoint";
            this.keyboardMultijointToolStripMenuItem.Click += new System.EventHandler(this.keyboardMultijointToolStripMenuItem_Click);
            // 
            // toolStripSeparator5
            // 
            this.toolStripSeparator5.Name = "toolStripSeparator5";
            this.toolStripSeparator5.Size = new System.Drawing.Size(217, 6);
            // 
            // AboutToolStripMenuItem
            // 
            this.AboutToolStripMenuItem.Name = "AboutToolStripMenuItem";
            this.AboutToolStripMenuItem.Size = new System.Drawing.Size(220, 26);
            this.AboutToolStripMenuItem.Text = "&About...";
            this.AboutToolStripMenuItem.Click += new System.EventHandler(this.AboutToolStripMenuItem_Click);
            // 
            // VoiceCoilCommBox
            // 
            this.VoiceCoilCommBox.Controls.Add(this.loadDLMButton);
            this.VoiceCoilCommBox.Controls.Add(this.model_name);
            this.VoiceCoilCommBox.Controls.Add(this.unloadButton);
            this.VoiceCoilCommBox.Controls.Add(this.Label3);
            this.VoiceCoilCommBox.Controls.Add(this.startButton);
            this.VoiceCoilCommBox.Controls.Add(this.loadButton);
            this.VoiceCoilCommBox.Controls.Add(this.stopButton);
            this.VoiceCoilCommBox.Controls.Add(this.disconnectButton);
            this.VoiceCoilCommBox.Controls.Add(this.Label9);
            this.VoiceCoilCommBox.Controls.Add(this.ipportTB);
            this.VoiceCoilCommBox.Controls.Add(this.ipaddressTB);
            this.VoiceCoilCommBox.Controls.Add(this.Label10);
            this.VoiceCoilCommBox.Controls.Add(this.connectButton);
            this.VoiceCoilCommBox.Location = new System.Drawing.Point(3, 6);
            this.VoiceCoilCommBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.VoiceCoilCommBox.Name = "VoiceCoilCommBox";
            this.VoiceCoilCommBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.VoiceCoilCommBox.Size = new System.Drawing.Size(1397, 63);
            this.VoiceCoilCommBox.TabIndex = 130;
            this.VoiceCoilCommBox.TabStop = false;
            this.VoiceCoilCommBox.Text = "xPC Target - Communication Settings";
            // 
            // loadDLMButton
            // 
            this.loadDLMButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.loadDLMButton.Location = new System.Drawing.Point(11, 26);
            this.loadDLMButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.loadDLMButton.Name = "loadDLMButton";
            this.loadDLMButton.Size = new System.Drawing.Size(116, 28);
            this.loadDLMButton.TabIndex = 10;
            this.loadDLMButton.Text = "Select Model...";
            this.loadDLMButton.Click += new System.EventHandler(this.loadDLMButton_Click_1);
            // 
            // model_name
            // 
            this.model_name.BackColor = System.Drawing.Color.White;
            this.model_name.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.model_name.Location = new System.Drawing.Point(240, 32);
            this.model_name.Name = "model_name";
            this.model_name.Size = new System.Drawing.Size(285, 18);
            this.model_name.TabIndex = 6;
            // 
            // unloadButton
            // 
            this.unloadButton.Enabled = false;
            this.unloadButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.unloadButton.Location = new System.Drawing.Point(1163, 26);
            this.unloadButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.unloadButton.Name = "unloadButton";
            this.unloadButton.Size = new System.Drawing.Size(67, 27);
            this.unloadButton.TabIndex = 9;
            this.unloadButton.Text = "Unload";
            this.unloadButton.Click += new System.EventHandler(this.unloadButton_Click_1);
            // 
            // Label3
            // 
            this.Label3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label3.Location = new System.Drawing.Point(152, 32);
            this.Label3.Name = "Label3";
            this.Label3.Size = new System.Drawing.Size(100, 28);
            this.Label3.TabIndex = 7;
            this.Label3.Text = "Model Name:";
            // 
            // startButton
            // 
            this.startButton.Enabled = false;
            this.startButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.startButton.Location = new System.Drawing.Point(1253, 26);
            this.startButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.startButton.Name = "startButton";
            this.startButton.Size = new System.Drawing.Size(53, 27);
            this.startButton.TabIndex = 8;
            this.startButton.Text = "Start";
            this.startButton.Click += new System.EventHandler(this.startButton_Click_1);
            // 
            // loadButton
            // 
            this.loadButton.Enabled = false;
            this.loadButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.loadButton.Location = new System.Drawing.Point(1097, 26);
            this.loadButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.loadButton.Name = "loadButton";
            this.loadButton.Size = new System.Drawing.Size(67, 27);
            this.loadButton.TabIndex = 8;
            this.loadButton.Text = "Load";
            this.loadButton.Click += new System.EventHandler(this.loadButton_Click_1);
            // 
            // stopButton
            // 
            this.stopButton.Enabled = false;
            this.stopButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.stopButton.Location = new System.Drawing.Point(1307, 26);
            this.stopButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.stopButton.Name = "stopButton";
            this.stopButton.Size = new System.Drawing.Size(53, 27);
            this.stopButton.TabIndex = 9;
            this.stopButton.Text = "Stop";
            this.stopButton.Click += new System.EventHandler(this.stopButton_Click_1);
            // 
            // disconnectButton
            // 
            this.disconnectButton.Enabled = false;
            this.disconnectButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.disconnectButton.Location = new System.Drawing.Point(987, 26);
            this.disconnectButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.disconnectButton.Name = "disconnectButton";
            this.disconnectButton.Size = new System.Drawing.Size(87, 27);
            this.disconnectButton.TabIndex = 7;
            this.disconnectButton.Text = "Disconnect";
            this.disconnectButton.Click += new System.EventHandler(this.disconnectButton_Click_1);
            // 
            // Label9
            // 
            this.Label9.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label9.Location = new System.Drawing.Point(559, 32);
            this.Label9.Name = "Label9";
            this.Label9.Size = new System.Drawing.Size(87, 15);
            this.Label9.TabIndex = 1;
            this.Label9.Text = "IP Address:";
            // 
            // ipportTB
            // 
            this.ipportTB.Location = new System.Drawing.Point(809, 30);
            this.ipportTB.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ipportTB.Name = "ipportTB";
            this.ipportTB.Size = new System.Drawing.Size(51, 22);
            this.ipportTB.TabIndex = 3;
            this.ipportTB.Text = "22222";
            // 
            // ipaddressTB
            // 
            this.ipaddressTB.Location = new System.Drawing.Point(645, 30);
            this.ipaddressTB.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ipaddressTB.Name = "ipaddressTB";
            this.ipaddressTB.Size = new System.Drawing.Size(99, 22);
            this.ipaddressTB.TabIndex = 0;
            this.ipaddressTB.Text = "129.128.14.90";
            // 
            // Label10
            // 
            this.Label10.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label10.Location = new System.Drawing.Point(755, 32);
            this.Label10.Name = "Label10";
            this.Label10.Size = new System.Drawing.Size(67, 15);
            this.Label10.TabIndex = 2;
            this.Label10.Text = "IP Port:";
            // 
            // connectButton
            // 
            this.connectButton.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.connectButton.Location = new System.Drawing.Point(899, 26);
            this.connectButton.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.connectButton.Name = "connectButton";
            this.connectButton.Size = new System.Drawing.Size(87, 27);
            this.connectButton.TabIndex = 1;
            this.connectButton.Text = "Connect";
            this.connectButton.Click += new System.EventHandler(this.connectButton_Click_1);
            // 
            // EMGParamBox
            // 
            this.EMGParamBox.Controls.Add(this.groupBox3);
            this.EMGParamBox.Controls.Add(this.groupBox1);
            this.EMGParamBox.Controls.Add(this.DoF2box);
            this.EMGParamBox.Controls.Add(this.DoF1box);
            this.EMGParamBox.Controls.Add(this.SwitchBox);
            this.EMGParamBox.Location = new System.Drawing.Point(4, 74);
            this.EMGParamBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.EMGParamBox.Name = "EMGParamBox";
            this.EMGParamBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.EMGParamBox.Size = new System.Drawing.Size(1397, 434);
            this.EMGParamBox.TabIndex = 136;
            this.EMGParamBox.TabStop = false;
            this.EMGParamBox.Text = "EMG Acquisition - Parameters";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.pictureBox7);
            this.groupBox3.Controls.Add(this.label78);
            this.groupBox3.Controls.Add(this.DoF4_mode_box);
            this.groupBox3.Controls.Add(this.pictureBox8);
            this.groupBox3.Controls.Add(this.label79);
            this.groupBox3.Controls.Add(this.ch8_smax_label);
            this.groupBox3.Controls.Add(this.DoF4_flip_checkBox);
            this.groupBox3.Controls.Add(this.ch8_smin_label);
            this.groupBox3.Controls.Add(this.ch7_smax_label);
            this.groupBox3.Controls.Add(this.label83);
            this.groupBox3.Controls.Add(this.ch7_smin_label);
            this.groupBox3.Controls.Add(this.ch8_smin_tick);
            this.groupBox3.Controls.Add(this.ch8_smax_tick);
            this.groupBox3.Controls.Add(this.ch7_smin_tick);
            this.groupBox3.Controls.Add(this.ch7_smax_tick);
            this.groupBox3.Controls.Add(this.ch8_smax_ctrl);
            this.groupBox3.Controls.Add(this.ch7_smax_ctrl);
            this.groupBox3.Controls.Add(this.label89);
            this.groupBox3.Controls.Add(this.ch8_smin_ctrl);
            this.groupBox3.Controls.Add(this.ch7_smin_ctrl);
            this.groupBox3.Controls.Add(this.label90);
            this.groupBox3.Controls.Add(this.label91);
            this.groupBox3.Controls.Add(this.ch8_gain_ctrl);
            this.groupBox3.Controls.Add(this.label92);
            this.groupBox3.Controls.Add(this.label93);
            this.groupBox3.Controls.Add(this.MAV7_bar);
            this.groupBox3.Controls.Add(this.DoF4_mapping_combobox);
            this.groupBox3.Controls.Add(this.label94);
            this.groupBox3.Controls.Add(this.label95);
            this.groupBox3.Controls.Add(this.MAV8_bar);
            this.groupBox3.Controls.Add(this.ch7_gain_ctrl);
            this.groupBox3.Controls.Add(this.label96);
            this.groupBox3.Location = new System.Drawing.Point(467, 226);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Size = new System.Drawing.Size(451, 199);
            this.groupBox3.TabIndex = 137;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Degree of Freedom 4 - Joystick";
            // 
            // pictureBox7
            // 
            this.pictureBox7.Image = global::brachIOplexus.Properties.Resources.bottom_arrow_rev2;
            this.pictureBox7.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox7.InitialImage")));
            this.pictureBox7.Location = new System.Drawing.Point(245, 137);
            this.pictureBox7.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox7.Name = "pictureBox7";
            this.pictureBox7.Size = new System.Drawing.Size(29, 30);
            this.pictureBox7.TabIndex = 134;
            this.pictureBox7.TabStop = false;
            // 
            // label78
            // 
            this.label78.AutoSize = true;
            this.label78.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label78.Location = new System.Drawing.Point(284, 30);
            this.label78.Name = "label78";
            this.label78.Size = new System.Drawing.Size(17, 17);
            this.label78.TabIndex = 127;
            this.label78.Text = "&&";
            // 
            // DoF4_mode_box
            // 
            this.DoF4_mode_box.DisplayMember = "1";
            this.DoF4_mode_box.FormattingEnabled = true;
            this.DoF4_mode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.DoF4_mode_box.Location = new System.Drawing.Point(301, 27);
            this.DoF4_mode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF4_mode_box.Name = "DoF4_mode_box";
            this.DoF4_mode_box.Size = new System.Drawing.Size(105, 24);
            this.DoF4_mode_box.TabIndex = 130;
            this.DoF4_mode_box.SelectedIndexChanged += new System.EventHandler(this.DoF4_mode_box_SelectedIndexChanged);
            // 
            // pictureBox8
            // 
            this.pictureBox8.Image = global::brachIOplexus.Properties.Resources.top_arrow_rev2;
            this.pictureBox8.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox8.InitialImage")));
            this.pictureBox8.Location = new System.Drawing.Point(245, 89);
            this.pictureBox8.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox8.Name = "pictureBox8";
            this.pictureBox8.Size = new System.Drawing.Size(29, 30);
            this.pictureBox8.TabIndex = 133;
            this.pictureBox8.TabStop = false;
            // 
            // label79
            // 
            this.label79.AutoSize = true;
            this.label79.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label79.Location = new System.Drawing.Point(83, 30);
            this.label79.Name = "label79";
            this.label79.Size = new System.Drawing.Size(66, 17);
            this.label79.TabIndex = 51;
            this.label79.Text = "Ch7/Ch8 ";
            // 
            // ch8_smax_label
            // 
            this.ch8_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch8_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch8_smax_label.Location = new System.Drawing.Point(211, 172);
            this.ch8_smax_label.Name = "ch8_smax_label";
            this.ch8_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch8_smax_label.TabIndex = 45;
            this.ch8_smax_label.Text = "Smax";
            // 
            // DoF4_flip_checkBox
            // 
            this.DoF4_flip_checkBox.AutoSize = true;
            this.DoF4_flip_checkBox.Location = new System.Drawing.Point(259, 121);
            this.DoF4_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF4_flip_checkBox.Name = "DoF4_flip_checkBox";
            this.DoF4_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.DoF4_flip_checkBox.TabIndex = 132;
            this.DoF4_flip_checkBox.UseVisualStyleBackColor = true;
            this.DoF4_flip_checkBox.CheckedChanged += new System.EventHandler(this.DoF4_flip_checkBox_CheckedChanged);
            // 
            // ch8_smin_label
            // 
            this.ch8_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch8_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch8_smin_label.Location = new System.Drawing.Point(36, 172);
            this.ch8_smin_label.Name = "ch8_smin_label";
            this.ch8_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch8_smin_label.TabIndex = 44;
            this.ch8_smin_label.Text = "Smin";
            // 
            // ch7_smax_label
            // 
            this.ch7_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch7_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch7_smax_label.Location = new System.Drawing.Point(211, 119);
            this.ch7_smax_label.Name = "ch7_smax_label";
            this.ch7_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch7_smax_label.TabIndex = 43;
            this.ch7_smax_label.Text = "Smax";
            // 
            // label83
            // 
            this.label83.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label83.Location = new System.Drawing.Point(243, 68);
            this.label83.Name = "label83";
            this.label83.Size = new System.Drawing.Size(35, 18);
            this.label83.TabIndex = 131;
            this.label83.Text = "Flip:";
            // 
            // ch7_smin_label
            // 
            this.ch7_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch7_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch7_smin_label.Location = new System.Drawing.Point(36, 119);
            this.ch7_smin_label.Name = "ch7_smin_label";
            this.ch7_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch7_smin_label.TabIndex = 42;
            this.ch7_smin_label.Text = "Smin";
            // 
            // ch8_smin_tick
            // 
            this.ch8_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch8_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch8_smin_tick.Location = new System.Drawing.Point(52, 143);
            this.ch8_smin_tick.Name = "ch8_smin_tick";
            this.ch8_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch8_smin_tick.TabIndex = 41;
            // 
            // ch8_smax_tick
            // 
            this.ch8_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch8_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch8_smax_tick.Location = new System.Drawing.Point(229, 143);
            this.ch8_smax_tick.Name = "ch8_smax_tick";
            this.ch8_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch8_smax_tick.TabIndex = 40;
            // 
            // ch7_smin_tick
            // 
            this.ch7_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch7_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch7_smin_tick.Location = new System.Drawing.Point(52, 90);
            this.ch7_smin_tick.Name = "ch7_smin_tick";
            this.ch7_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch7_smin_tick.TabIndex = 39;
            // 
            // ch7_smax_tick
            // 
            this.ch7_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch7_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch7_smax_tick.Location = new System.Drawing.Point(229, 90);
            this.ch7_smax_tick.Name = "ch7_smax_tick";
            this.ch7_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch7_smax_tick.TabIndex = 38;
            // 
            // ch8_smax_ctrl
            // 
            this.ch8_smax_ctrl.DecimalPlaces = 1;
            this.ch8_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch8_smax_ctrl.Location = new System.Drawing.Point(399, 145);
            this.ch8_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch8_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch8_smax_ctrl.Name = "ch8_smax_ctrl";
            this.ch8_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch8_smax_ctrl.TabIndex = 31;
            this.ch8_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch8_smax_ctrl.ValueChanged += new System.EventHandler(this.ch8_smax_ctrl_ValueChanged);
            // 
            // ch7_smax_ctrl
            // 
            this.ch7_smax_ctrl.DecimalPlaces = 1;
            this.ch7_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch7_smax_ctrl.Location = new System.Drawing.Point(399, 90);
            this.ch7_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch7_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch7_smax_ctrl.Name = "ch7_smax_ctrl";
            this.ch7_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch7_smax_ctrl.TabIndex = 30;
            this.ch7_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch7_smax_ctrl.ValueChanged += new System.EventHandler(this.ch7_smax_ctrl_ValueChanged);
            // 
            // label89
            // 
            this.label89.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label89.Location = new System.Drawing.Point(395, 68);
            this.label89.Name = "label89";
            this.label89.Size = new System.Drawing.Size(47, 18);
            this.label89.TabIndex = 29;
            this.label89.Text = "Smax:";
            // 
            // ch8_smin_ctrl
            // 
            this.ch8_smin_ctrl.DecimalPlaces = 1;
            this.ch8_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch8_smin_ctrl.Location = new System.Drawing.Point(349, 145);
            this.ch8_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch8_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch8_smin_ctrl.Name = "ch8_smin_ctrl";
            this.ch8_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch8_smin_ctrl.TabIndex = 28;
            this.ch8_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch8_smin_ctrl.ValueChanged += new System.EventHandler(this.ch8_smin_ctrl_ValueChanged);
            // 
            // ch7_smin_ctrl
            // 
            this.ch7_smin_ctrl.DecimalPlaces = 1;
            this.ch7_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch7_smin_ctrl.Location = new System.Drawing.Point(349, 90);
            this.ch7_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch7_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch7_smin_ctrl.Name = "ch7_smin_ctrl";
            this.ch7_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch7_smin_ctrl.TabIndex = 27;
            this.ch7_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch7_smin_ctrl.ValueChanged += new System.EventHandler(this.ch7_smin_ctrl_ValueChanged);
            // 
            // label90
            // 
            this.label90.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label90.Location = new System.Drawing.Point(347, 68);
            this.label90.Name = "label90";
            this.label90.Size = new System.Drawing.Size(47, 18);
            this.label90.TabIndex = 26;
            this.label90.Text = "Smin:";
            // 
            // label91
            // 
            this.label91.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label91.Location = new System.Drawing.Point(49, 68);
            this.label91.Name = "label91";
            this.label91.Size = new System.Drawing.Size(105, 18);
            this.label91.TabIndex = 25;
            this.label91.Text = "Signal Strength:";
            // 
            // ch8_gain_ctrl
            // 
            this.ch8_gain_ctrl.Location = new System.Drawing.Point(285, 145);
            this.ch8_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch8_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch8_gain_ctrl.Name = "ch8_gain_ctrl";
            this.ch8_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch8_gain_ctrl.TabIndex = 24;
            this.ch8_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch8_gain_ctrl.ValueChanged += new System.EventHandler(this.ch8_gain_ctrl_ValueChanged);
            // 
            // label92
            // 
            this.label92.AutoSize = true;
            this.label92.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label92.Location = new System.Drawing.Point(13, 145);
            this.label92.Name = "label92";
            this.label92.Size = new System.Drawing.Size(37, 17);
            this.label92.TabIndex = 23;
            this.label92.Text = "Ch8:";
            // 
            // label93
            // 
            this.label93.AutoSize = true;
            this.label93.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label93.Location = new System.Drawing.Point(12, 90);
            this.label93.Name = "label93";
            this.label93.Size = new System.Drawing.Size(37, 17);
            this.label93.TabIndex = 22;
            this.label93.Text = "Ch7:";
            // 
            // MAV7_bar
            // 
            this.MAV7_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV7_bar.Location = new System.Drawing.Point(52, 90);
            this.MAV7_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV7_bar.MarqueeAnimationSpeed = 30;
            this.MAV7_bar.Maximum = 500;
            this.MAV7_bar.Name = "MAV7_bar";
            this.MAV7_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV7_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV7_bar.TabIndex = 16;
            this.MAV7_bar.Value = 200;
            // 
            // DoF4_mapping_combobox
            // 
            this.DoF4_mapping_combobox.FormattingEnabled = true;
            this.DoF4_mapping_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.DoF4_mapping_combobox.Location = new System.Drawing.Point(175, 27);
            this.DoF4_mapping_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF4_mapping_combobox.Name = "DoF4_mapping_combobox";
            this.DoF4_mapping_combobox.Size = new System.Drawing.Size(105, 24);
            this.DoF4_mapping_combobox.TabIndex = 21;
            this.DoF4_mapping_combobox.SelectedIndexChanged += new System.EventHandler(this.DoF4_mapping_combobox_SelectedIndexChanged);
            // 
            // label94
            // 
            this.label94.AutoSize = true;
            this.label94.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label94.Location = new System.Drawing.Point(147, 30);
            this.label94.Name = "label94";
            this.label94.Size = new System.Drawing.Size(24, 17);
            this.label94.TabIndex = 20;
            this.label94.Text = ">>";
            // 
            // label95
            // 
            this.label95.AutoSize = true;
            this.label95.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label95.Location = new System.Drawing.Point(13, 30);
            this.label95.Name = "label95";
            this.label95.Size = new System.Drawing.Size(66, 17);
            this.label95.TabIndex = 18;
            this.label95.Text = "Mapping:";
            // 
            // MAV8_bar
            // 
            this.MAV8_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV8_bar.Location = new System.Drawing.Point(52, 143);
            this.MAV8_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV8_bar.MarqueeAnimationSpeed = 30;
            this.MAV8_bar.Maximum = 500;
            this.MAV8_bar.Name = "MAV8_bar";
            this.MAV8_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV8_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV8_bar.TabIndex = 17;
            this.MAV8_bar.Value = 200;
            // 
            // ch7_gain_ctrl
            // 
            this.ch7_gain_ctrl.Location = new System.Drawing.Point(285, 90);
            this.ch7_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch7_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch7_gain_ctrl.Name = "ch7_gain_ctrl";
            this.ch7_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch7_gain_ctrl.TabIndex = 15;
            this.ch7_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch7_gain_ctrl.ValueChanged += new System.EventHandler(this.ch7_gain_ctrl_ValueChanged);
            // 
            // label96
            // 
            this.label96.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label96.Location = new System.Drawing.Point(283, 68);
            this.label96.Name = "label96";
            this.label96.Size = new System.Drawing.Size(77, 18);
            this.label96.TabIndex = 14;
            this.label96.Text = "Gain:";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.pictureBox5);
            this.groupBox1.Controls.Add(this.label51);
            this.groupBox1.Controls.Add(this.DoF3_mode_box);
            this.groupBox1.Controls.Add(this.pictureBox6);
            this.groupBox1.Controls.Add(this.label52);
            this.groupBox1.Controls.Add(this.ch6_smax_label);
            this.groupBox1.Controls.Add(this.DoF3_flip_checkBox);
            this.groupBox1.Controls.Add(this.ch6_smin_label);
            this.groupBox1.Controls.Add(this.ch5_smax_label);
            this.groupBox1.Controls.Add(this.label57);
            this.groupBox1.Controls.Add(this.ch5_smin_label);
            this.groupBox1.Controls.Add(this.ch6_smin_tick);
            this.groupBox1.Controls.Add(this.ch6_smax_tick);
            this.groupBox1.Controls.Add(this.ch5_smin_tick);
            this.groupBox1.Controls.Add(this.ch5_smax_tick);
            this.groupBox1.Controls.Add(this.ch6_smax_ctrl);
            this.groupBox1.Controls.Add(this.ch5_smax_ctrl);
            this.groupBox1.Controls.Add(this.label63);
            this.groupBox1.Controls.Add(this.ch6_smin_ctrl);
            this.groupBox1.Controls.Add(this.ch5_smin_ctrl);
            this.groupBox1.Controls.Add(this.label64);
            this.groupBox1.Controls.Add(this.label65);
            this.groupBox1.Controls.Add(this.ch6_gain_ctrl);
            this.groupBox1.Controls.Add(this.label66);
            this.groupBox1.Controls.Add(this.label67);
            this.groupBox1.Controls.Add(this.MAV5_bar);
            this.groupBox1.Controls.Add(this.DoF3_mapping_combobox);
            this.groupBox1.Controls.Add(this.label68);
            this.groupBox1.Controls.Add(this.label72);
            this.groupBox1.Controls.Add(this.MAV6_bar);
            this.groupBox1.Controls.Add(this.ch5_gain_ctrl);
            this.groupBox1.Controls.Add(this.label77);
            this.groupBox1.Location = new System.Drawing.Point(467, 21);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox1.Size = new System.Drawing.Size(451, 199);
            this.groupBox1.TabIndex = 136;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Degree of Freedom 3 - Joystick";
            // 
            // pictureBox5
            // 
            this.pictureBox5.Image = global::brachIOplexus.Properties.Resources.bottom_arrow_rev2;
            this.pictureBox5.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox5.InitialImage")));
            this.pictureBox5.Location = new System.Drawing.Point(245, 137);
            this.pictureBox5.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox5.Name = "pictureBox5";
            this.pictureBox5.Size = new System.Drawing.Size(29, 30);
            this.pictureBox5.TabIndex = 134;
            this.pictureBox5.TabStop = false;
            // 
            // label51
            // 
            this.label51.AutoSize = true;
            this.label51.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label51.Location = new System.Drawing.Point(284, 30);
            this.label51.Name = "label51";
            this.label51.Size = new System.Drawing.Size(17, 17);
            this.label51.TabIndex = 127;
            this.label51.Text = "&&";
            // 
            // DoF3_mode_box
            // 
            this.DoF3_mode_box.DisplayMember = "1";
            this.DoF3_mode_box.FormattingEnabled = true;
            this.DoF3_mode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.DoF3_mode_box.Location = new System.Drawing.Point(301, 27);
            this.DoF3_mode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF3_mode_box.Name = "DoF3_mode_box";
            this.DoF3_mode_box.Size = new System.Drawing.Size(105, 24);
            this.DoF3_mode_box.TabIndex = 130;
            this.DoF3_mode_box.SelectedIndexChanged += new System.EventHandler(this.DoF3_mode_box_SelectedIndexChanged);
            // 
            // pictureBox6
            // 
            this.pictureBox6.Image = global::brachIOplexus.Properties.Resources.top_arrow_rev2;
            this.pictureBox6.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox6.InitialImage")));
            this.pictureBox6.Location = new System.Drawing.Point(245, 89);
            this.pictureBox6.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox6.Name = "pictureBox6";
            this.pictureBox6.Size = new System.Drawing.Size(29, 30);
            this.pictureBox6.TabIndex = 133;
            this.pictureBox6.TabStop = false;
            // 
            // label52
            // 
            this.label52.AutoSize = true;
            this.label52.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label52.Location = new System.Drawing.Point(83, 30);
            this.label52.Name = "label52";
            this.label52.Size = new System.Drawing.Size(66, 17);
            this.label52.TabIndex = 51;
            this.label52.Text = "Ch5/Ch6 ";
            // 
            // ch6_smax_label
            // 
            this.ch6_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch6_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch6_smax_label.Location = new System.Drawing.Point(211, 172);
            this.ch6_smax_label.Name = "ch6_smax_label";
            this.ch6_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch6_smax_label.TabIndex = 45;
            this.ch6_smax_label.Text = "Smax";
            // 
            // DoF3_flip_checkBox
            // 
            this.DoF3_flip_checkBox.AutoSize = true;
            this.DoF3_flip_checkBox.Location = new System.Drawing.Point(259, 121);
            this.DoF3_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF3_flip_checkBox.Name = "DoF3_flip_checkBox";
            this.DoF3_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.DoF3_flip_checkBox.TabIndex = 132;
            this.DoF3_flip_checkBox.UseVisualStyleBackColor = true;
            this.DoF3_flip_checkBox.CheckedChanged += new System.EventHandler(this.DoF3_flip_checkBox_CheckedChanged);
            // 
            // ch6_smin_label
            // 
            this.ch6_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch6_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch6_smin_label.Location = new System.Drawing.Point(36, 172);
            this.ch6_smin_label.Name = "ch6_smin_label";
            this.ch6_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch6_smin_label.TabIndex = 44;
            this.ch6_smin_label.Text = "Smin";
            // 
            // ch5_smax_label
            // 
            this.ch5_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch5_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch5_smax_label.Location = new System.Drawing.Point(211, 119);
            this.ch5_smax_label.Name = "ch5_smax_label";
            this.ch5_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch5_smax_label.TabIndex = 43;
            this.ch5_smax_label.Text = "Smax";
            // 
            // label57
            // 
            this.label57.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label57.Location = new System.Drawing.Point(243, 68);
            this.label57.Name = "label57";
            this.label57.Size = new System.Drawing.Size(35, 18);
            this.label57.TabIndex = 131;
            this.label57.Text = "Flip:";
            // 
            // ch5_smin_label
            // 
            this.ch5_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch5_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch5_smin_label.Location = new System.Drawing.Point(36, 119);
            this.ch5_smin_label.Name = "ch5_smin_label";
            this.ch5_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch5_smin_label.TabIndex = 42;
            this.ch5_smin_label.Text = "Smin";
            // 
            // ch6_smin_tick
            // 
            this.ch6_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch6_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch6_smin_tick.Location = new System.Drawing.Point(52, 143);
            this.ch6_smin_tick.Name = "ch6_smin_tick";
            this.ch6_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch6_smin_tick.TabIndex = 41;
            // 
            // ch6_smax_tick
            // 
            this.ch6_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch6_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch6_smax_tick.Location = new System.Drawing.Point(229, 143);
            this.ch6_smax_tick.Name = "ch6_smax_tick";
            this.ch6_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch6_smax_tick.TabIndex = 40;
            // 
            // ch5_smin_tick
            // 
            this.ch5_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch5_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch5_smin_tick.Location = new System.Drawing.Point(52, 90);
            this.ch5_smin_tick.Name = "ch5_smin_tick";
            this.ch5_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch5_smin_tick.TabIndex = 39;
            // 
            // ch5_smax_tick
            // 
            this.ch5_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch5_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch5_smax_tick.Location = new System.Drawing.Point(229, 90);
            this.ch5_smax_tick.Name = "ch5_smax_tick";
            this.ch5_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch5_smax_tick.TabIndex = 38;
            // 
            // ch6_smax_ctrl
            // 
            this.ch6_smax_ctrl.DecimalPlaces = 1;
            this.ch6_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch6_smax_ctrl.Location = new System.Drawing.Point(399, 145);
            this.ch6_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch6_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch6_smax_ctrl.Name = "ch6_smax_ctrl";
            this.ch6_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch6_smax_ctrl.TabIndex = 31;
            this.ch6_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch6_smax_ctrl.ValueChanged += new System.EventHandler(this.ch6_smax_ctrl_ValueChanged);
            // 
            // ch5_smax_ctrl
            // 
            this.ch5_smax_ctrl.DecimalPlaces = 1;
            this.ch5_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch5_smax_ctrl.Location = new System.Drawing.Point(399, 90);
            this.ch5_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch5_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch5_smax_ctrl.Name = "ch5_smax_ctrl";
            this.ch5_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch5_smax_ctrl.TabIndex = 30;
            this.ch5_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch5_smax_ctrl.ValueChanged += new System.EventHandler(this.ch5_smax_ctrl_ValueChanged);
            // 
            // label63
            // 
            this.label63.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label63.Location = new System.Drawing.Point(395, 68);
            this.label63.Name = "label63";
            this.label63.Size = new System.Drawing.Size(47, 18);
            this.label63.TabIndex = 29;
            this.label63.Text = "Smax:";
            // 
            // ch6_smin_ctrl
            // 
            this.ch6_smin_ctrl.DecimalPlaces = 1;
            this.ch6_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch6_smin_ctrl.Location = new System.Drawing.Point(349, 145);
            this.ch6_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch6_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch6_smin_ctrl.Name = "ch6_smin_ctrl";
            this.ch6_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch6_smin_ctrl.TabIndex = 28;
            this.ch6_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch6_smin_ctrl.ValueChanged += new System.EventHandler(this.ch6_smin_ctrl_ValueChanged);
            // 
            // ch5_smin_ctrl
            // 
            this.ch5_smin_ctrl.DecimalPlaces = 1;
            this.ch5_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch5_smin_ctrl.Location = new System.Drawing.Point(349, 90);
            this.ch5_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch5_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch5_smin_ctrl.Name = "ch5_smin_ctrl";
            this.ch5_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch5_smin_ctrl.TabIndex = 27;
            this.ch5_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch5_smin_ctrl.ValueChanged += new System.EventHandler(this.ch5_smin_ctrl_ValueChanged);
            // 
            // label64
            // 
            this.label64.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label64.Location = new System.Drawing.Point(347, 68);
            this.label64.Name = "label64";
            this.label64.Size = new System.Drawing.Size(47, 18);
            this.label64.TabIndex = 26;
            this.label64.Text = "Smin:";
            // 
            // label65
            // 
            this.label65.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label65.Location = new System.Drawing.Point(49, 68);
            this.label65.Name = "label65";
            this.label65.Size = new System.Drawing.Size(105, 18);
            this.label65.TabIndex = 25;
            this.label65.Text = "Signal Strength:";
            // 
            // ch6_gain_ctrl
            // 
            this.ch6_gain_ctrl.Location = new System.Drawing.Point(285, 145);
            this.ch6_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch6_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch6_gain_ctrl.Name = "ch6_gain_ctrl";
            this.ch6_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch6_gain_ctrl.TabIndex = 24;
            this.ch6_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch6_gain_ctrl.ValueChanged += new System.EventHandler(this.ch6_gain_ctrl_ValueChanged);
            // 
            // label66
            // 
            this.label66.AutoSize = true;
            this.label66.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label66.Location = new System.Drawing.Point(13, 145);
            this.label66.Name = "label66";
            this.label66.Size = new System.Drawing.Size(37, 17);
            this.label66.TabIndex = 23;
            this.label66.Text = "Ch6:";
            // 
            // label67
            // 
            this.label67.AutoSize = true;
            this.label67.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label67.Location = new System.Drawing.Point(12, 90);
            this.label67.Name = "label67";
            this.label67.Size = new System.Drawing.Size(37, 17);
            this.label67.TabIndex = 22;
            this.label67.Text = "Ch5:";
            // 
            // MAV5_bar
            // 
            this.MAV5_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV5_bar.Location = new System.Drawing.Point(52, 90);
            this.MAV5_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV5_bar.MarqueeAnimationSpeed = 30;
            this.MAV5_bar.Maximum = 500;
            this.MAV5_bar.Name = "MAV5_bar";
            this.MAV5_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV5_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV5_bar.TabIndex = 16;
            this.MAV5_bar.Value = 200;
            // 
            // DoF3_mapping_combobox
            // 
            this.DoF3_mapping_combobox.FormattingEnabled = true;
            this.DoF3_mapping_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.DoF3_mapping_combobox.Location = new System.Drawing.Point(175, 27);
            this.DoF3_mapping_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF3_mapping_combobox.Name = "DoF3_mapping_combobox";
            this.DoF3_mapping_combobox.Size = new System.Drawing.Size(105, 24);
            this.DoF3_mapping_combobox.TabIndex = 21;
            this.DoF3_mapping_combobox.SelectedIndexChanged += new System.EventHandler(this.DoF3_mapping_combobox_SelectedIndexChanged);
            // 
            // label68
            // 
            this.label68.AutoSize = true;
            this.label68.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label68.Location = new System.Drawing.Point(147, 30);
            this.label68.Name = "label68";
            this.label68.Size = new System.Drawing.Size(24, 17);
            this.label68.TabIndex = 20;
            this.label68.Text = ">>";
            // 
            // label72
            // 
            this.label72.AutoSize = true;
            this.label72.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label72.Location = new System.Drawing.Point(13, 30);
            this.label72.Name = "label72";
            this.label72.Size = new System.Drawing.Size(66, 17);
            this.label72.TabIndex = 18;
            this.label72.Text = "Mapping:";
            // 
            // MAV6_bar
            // 
            this.MAV6_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV6_bar.Location = new System.Drawing.Point(52, 143);
            this.MAV6_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV6_bar.MarqueeAnimationSpeed = 30;
            this.MAV6_bar.Maximum = 500;
            this.MAV6_bar.Name = "MAV6_bar";
            this.MAV6_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV6_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV6_bar.TabIndex = 17;
            this.MAV6_bar.Value = 200;
            // 
            // ch5_gain_ctrl
            // 
            this.ch5_gain_ctrl.Location = new System.Drawing.Point(285, 90);
            this.ch5_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch5_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch5_gain_ctrl.Name = "ch5_gain_ctrl";
            this.ch5_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch5_gain_ctrl.TabIndex = 15;
            this.ch5_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch5_gain_ctrl.ValueChanged += new System.EventHandler(this.ch5_gain_ctrl_ValueChanged);
            // 
            // label77
            // 
            this.label77.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label77.Location = new System.Drawing.Point(283, 68);
            this.label77.Name = "label77";
            this.label77.Size = new System.Drawing.Size(77, 18);
            this.label77.TabIndex = 14;
            this.label77.Text = "Gain:";
            // 
            // DoF2box
            // 
            this.DoF2box.Controls.Add(this.pictureBox3);
            this.DoF2box.Controls.Add(this.label26);
            this.DoF2box.Controls.Add(this.DoF2_mode_box);
            this.DoF2box.Controls.Add(this.pictureBox4);
            this.DoF2box.Controls.Add(this.label28);
            this.DoF2box.Controls.Add(this.ch4_smax_label);
            this.DoF2box.Controls.Add(this.DoF2_flip_checkBox);
            this.DoF2box.Controls.Add(this.ch4_smin_label);
            this.DoF2box.Controls.Add(this.ch3_smax_label);
            this.DoF2box.Controls.Add(this.label33);
            this.DoF2box.Controls.Add(this.ch3_smin_label);
            this.DoF2box.Controls.Add(this.ch4_smin_tick);
            this.DoF2box.Controls.Add(this.ch4_smax_tick);
            this.DoF2box.Controls.Add(this.ch3_smin_tick);
            this.DoF2box.Controls.Add(this.ch3_smax_tick);
            this.DoF2box.Controls.Add(this.ch4_smax_ctrl);
            this.DoF2box.Controls.Add(this.ch3_smax_ctrl);
            this.DoF2box.Controls.Add(this.label43);
            this.DoF2box.Controls.Add(this.ch4_smin_ctrl);
            this.DoF2box.Controls.Add(this.ch3_smin_ctrl);
            this.DoF2box.Controls.Add(this.label44);
            this.DoF2box.Controls.Add(this.label45);
            this.DoF2box.Controls.Add(this.ch4_gain_ctrl);
            this.DoF2box.Controls.Add(this.label46);
            this.DoF2box.Controls.Add(this.label47);
            this.DoF2box.Controls.Add(this.MAV3_bar);
            this.DoF2box.Controls.Add(this.DoF2_mapping_combobox);
            this.DoF2box.Controls.Add(this.label48);
            this.DoF2box.Controls.Add(this.label49);
            this.DoF2box.Controls.Add(this.MAV4_bar);
            this.DoF2box.Controls.Add(this.ch3_gain_ctrl);
            this.DoF2box.Controls.Add(this.label50);
            this.DoF2box.Location = new System.Drawing.Point(11, 226);
            this.DoF2box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF2box.Name = "DoF2box";
            this.DoF2box.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF2box.Size = new System.Drawing.Size(451, 199);
            this.DoF2box.TabIndex = 135;
            this.DoF2box.TabStop = false;
            this.DoF2box.Text = "Degree of Freedom 2 - EMG (Upperarm)";
            // 
            // pictureBox3
            // 
            this.pictureBox3.Image = global::brachIOplexus.Properties.Resources.bottom_arrow_rev2;
            this.pictureBox3.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox3.InitialImage")));
            this.pictureBox3.Location = new System.Drawing.Point(245, 137);
            this.pictureBox3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(29, 30);
            this.pictureBox3.TabIndex = 134;
            this.pictureBox3.TabStop = false;
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label26.Location = new System.Drawing.Point(284, 30);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(17, 17);
            this.label26.TabIndex = 127;
            this.label26.Text = "&&";
            // 
            // DoF2_mode_box
            // 
            this.DoF2_mode_box.DisplayMember = "1";
            this.DoF2_mode_box.FormattingEnabled = true;
            this.DoF2_mode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.DoF2_mode_box.Location = new System.Drawing.Point(301, 27);
            this.DoF2_mode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF2_mode_box.Name = "DoF2_mode_box";
            this.DoF2_mode_box.Size = new System.Drawing.Size(105, 24);
            this.DoF2_mode_box.TabIndex = 130;
            this.DoF2_mode_box.SelectedIndexChanged += new System.EventHandler(this.DoF2_mode_box_SelectedIndexChanged);
            // 
            // pictureBox4
            // 
            this.pictureBox4.Image = global::brachIOplexus.Properties.Resources.top_arrow_rev2;
            this.pictureBox4.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox4.InitialImage")));
            this.pictureBox4.Location = new System.Drawing.Point(245, 89);
            this.pictureBox4.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox4.Name = "pictureBox4";
            this.pictureBox4.Size = new System.Drawing.Size(29, 30);
            this.pictureBox4.TabIndex = 133;
            this.pictureBox4.TabStop = false;
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label28.Location = new System.Drawing.Point(83, 30);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(66, 17);
            this.label28.TabIndex = 51;
            this.label28.Text = "Ch3/Ch4 ";
            // 
            // ch4_smax_label
            // 
            this.ch4_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch4_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch4_smax_label.Location = new System.Drawing.Point(211, 172);
            this.ch4_smax_label.Name = "ch4_smax_label";
            this.ch4_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch4_smax_label.TabIndex = 45;
            this.ch4_smax_label.Text = "Smax";
            // 
            // DoF2_flip_checkBox
            // 
            this.DoF2_flip_checkBox.AutoSize = true;
            this.DoF2_flip_checkBox.Location = new System.Drawing.Point(259, 121);
            this.DoF2_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF2_flip_checkBox.Name = "DoF2_flip_checkBox";
            this.DoF2_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.DoF2_flip_checkBox.TabIndex = 132;
            this.DoF2_flip_checkBox.UseVisualStyleBackColor = true;
            this.DoF2_flip_checkBox.CheckedChanged += new System.EventHandler(this.DoF2_flip_checkBox_CheckedChanged);
            // 
            // ch4_smin_label
            // 
            this.ch4_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch4_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch4_smin_label.Location = new System.Drawing.Point(36, 172);
            this.ch4_smin_label.Name = "ch4_smin_label";
            this.ch4_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch4_smin_label.TabIndex = 44;
            this.ch4_smin_label.Text = "Smin";
            // 
            // ch3_smax_label
            // 
            this.ch3_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch3_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch3_smax_label.Location = new System.Drawing.Point(211, 119);
            this.ch3_smax_label.Name = "ch3_smax_label";
            this.ch3_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch3_smax_label.TabIndex = 43;
            this.ch3_smax_label.Text = "Smax";
            // 
            // label33
            // 
            this.label33.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label33.Location = new System.Drawing.Point(243, 68);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(35, 18);
            this.label33.TabIndex = 131;
            this.label33.Text = "Flip:";
            // 
            // ch3_smin_label
            // 
            this.ch3_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch3_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch3_smin_label.Location = new System.Drawing.Point(36, 119);
            this.ch3_smin_label.Name = "ch3_smin_label";
            this.ch3_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch3_smin_label.TabIndex = 42;
            this.ch3_smin_label.Text = "Smin";
            // 
            // ch4_smin_tick
            // 
            this.ch4_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch4_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch4_smin_tick.Location = new System.Drawing.Point(52, 143);
            this.ch4_smin_tick.Name = "ch4_smin_tick";
            this.ch4_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch4_smin_tick.TabIndex = 41;
            // 
            // ch4_smax_tick
            // 
            this.ch4_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch4_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch4_smax_tick.Location = new System.Drawing.Point(229, 143);
            this.ch4_smax_tick.Name = "ch4_smax_tick";
            this.ch4_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch4_smax_tick.TabIndex = 40;
            // 
            // ch3_smin_tick
            // 
            this.ch3_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch3_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch3_smin_tick.Location = new System.Drawing.Point(52, 90);
            this.ch3_smin_tick.Name = "ch3_smin_tick";
            this.ch3_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch3_smin_tick.TabIndex = 39;
            // 
            // ch3_smax_tick
            // 
            this.ch3_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch3_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch3_smax_tick.Location = new System.Drawing.Point(229, 90);
            this.ch3_smax_tick.Name = "ch3_smax_tick";
            this.ch3_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch3_smax_tick.TabIndex = 38;
            // 
            // ch4_smax_ctrl
            // 
            this.ch4_smax_ctrl.DecimalPlaces = 1;
            this.ch4_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch4_smax_ctrl.Location = new System.Drawing.Point(399, 145);
            this.ch4_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch4_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch4_smax_ctrl.Name = "ch4_smax_ctrl";
            this.ch4_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch4_smax_ctrl.TabIndex = 31;
            this.ch4_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch4_smax_ctrl.ValueChanged += new System.EventHandler(this.ch4_smax_ctrl_ValueChanged);
            // 
            // ch3_smax_ctrl
            // 
            this.ch3_smax_ctrl.DecimalPlaces = 1;
            this.ch3_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch3_smax_ctrl.Location = new System.Drawing.Point(399, 90);
            this.ch3_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch3_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch3_smax_ctrl.Name = "ch3_smax_ctrl";
            this.ch3_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch3_smax_ctrl.TabIndex = 30;
            this.ch3_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch3_smax_ctrl.ValueChanged += new System.EventHandler(this.ch3_smax_ctrl_ValueChanged);
            // 
            // label43
            // 
            this.label43.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label43.Location = new System.Drawing.Point(395, 68);
            this.label43.Name = "label43";
            this.label43.Size = new System.Drawing.Size(47, 18);
            this.label43.TabIndex = 29;
            this.label43.Text = "Smax:";
            // 
            // ch4_smin_ctrl
            // 
            this.ch4_smin_ctrl.DecimalPlaces = 1;
            this.ch4_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch4_smin_ctrl.Location = new System.Drawing.Point(349, 145);
            this.ch4_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch4_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch4_smin_ctrl.Name = "ch4_smin_ctrl";
            this.ch4_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch4_smin_ctrl.TabIndex = 28;
            this.ch4_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch4_smin_ctrl.ValueChanged += new System.EventHandler(this.ch4_smin_ctrl_ValueChanged);
            // 
            // ch3_smin_ctrl
            // 
            this.ch3_smin_ctrl.DecimalPlaces = 1;
            this.ch3_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch3_smin_ctrl.Location = new System.Drawing.Point(349, 90);
            this.ch3_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch3_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch3_smin_ctrl.Name = "ch3_smin_ctrl";
            this.ch3_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch3_smin_ctrl.TabIndex = 27;
            this.ch3_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch3_smin_ctrl.ValueChanged += new System.EventHandler(this.ch3_smin_ctrl_ValueChanged);
            // 
            // label44
            // 
            this.label44.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label44.Location = new System.Drawing.Point(347, 68);
            this.label44.Name = "label44";
            this.label44.Size = new System.Drawing.Size(47, 18);
            this.label44.TabIndex = 26;
            this.label44.Text = "Smin:";
            // 
            // label45
            // 
            this.label45.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label45.Location = new System.Drawing.Point(49, 68);
            this.label45.Name = "label45";
            this.label45.Size = new System.Drawing.Size(105, 18);
            this.label45.TabIndex = 25;
            this.label45.Text = "Signal Strength:";
            // 
            // ch4_gain_ctrl
            // 
            this.ch4_gain_ctrl.Location = new System.Drawing.Point(285, 145);
            this.ch4_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch4_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch4_gain_ctrl.Name = "ch4_gain_ctrl";
            this.ch4_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch4_gain_ctrl.TabIndex = 24;
            this.ch4_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch4_gain_ctrl.ValueChanged += new System.EventHandler(this.ch4_gain_ctrl_ValueChanged);
            // 
            // label46
            // 
            this.label46.AutoSize = true;
            this.label46.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label46.Location = new System.Drawing.Point(13, 145);
            this.label46.Name = "label46";
            this.label46.Size = new System.Drawing.Size(37, 17);
            this.label46.TabIndex = 23;
            this.label46.Text = "Ch4:";
            // 
            // label47
            // 
            this.label47.AutoSize = true;
            this.label47.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label47.Location = new System.Drawing.Point(12, 90);
            this.label47.Name = "label47";
            this.label47.Size = new System.Drawing.Size(37, 17);
            this.label47.TabIndex = 22;
            this.label47.Text = "Ch3:";
            // 
            // MAV3_bar
            // 
            this.MAV3_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV3_bar.Location = new System.Drawing.Point(52, 90);
            this.MAV3_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV3_bar.MarqueeAnimationSpeed = 30;
            this.MAV3_bar.Maximum = 500;
            this.MAV3_bar.Name = "MAV3_bar";
            this.MAV3_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV3_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV3_bar.TabIndex = 16;
            this.MAV3_bar.Value = 200;
            // 
            // DoF2_mapping_combobox
            // 
            this.DoF2_mapping_combobox.FormattingEnabled = true;
            this.DoF2_mapping_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.DoF2_mapping_combobox.Location = new System.Drawing.Point(175, 27);
            this.DoF2_mapping_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF2_mapping_combobox.Name = "DoF2_mapping_combobox";
            this.DoF2_mapping_combobox.Size = new System.Drawing.Size(105, 24);
            this.DoF2_mapping_combobox.TabIndex = 21;
            this.DoF2_mapping_combobox.SelectedIndexChanged += new System.EventHandler(this.DoF2_mapping_combobox_SelectedIndexChanged);
            // 
            // label48
            // 
            this.label48.AutoSize = true;
            this.label48.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label48.Location = new System.Drawing.Point(147, 30);
            this.label48.Name = "label48";
            this.label48.Size = new System.Drawing.Size(24, 17);
            this.label48.TabIndex = 20;
            this.label48.Text = ">>";
            // 
            // label49
            // 
            this.label49.AutoSize = true;
            this.label49.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label49.Location = new System.Drawing.Point(13, 30);
            this.label49.Name = "label49";
            this.label49.Size = new System.Drawing.Size(66, 17);
            this.label49.TabIndex = 18;
            this.label49.Text = "Mapping:";
            // 
            // MAV4_bar
            // 
            this.MAV4_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV4_bar.Location = new System.Drawing.Point(52, 143);
            this.MAV4_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV4_bar.MarqueeAnimationSpeed = 30;
            this.MAV4_bar.Maximum = 500;
            this.MAV4_bar.Name = "MAV4_bar";
            this.MAV4_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV4_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV4_bar.TabIndex = 17;
            this.MAV4_bar.Value = 200;
            // 
            // ch3_gain_ctrl
            // 
            this.ch3_gain_ctrl.Location = new System.Drawing.Point(285, 90);
            this.ch3_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch3_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch3_gain_ctrl.Name = "ch3_gain_ctrl";
            this.ch3_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch3_gain_ctrl.TabIndex = 15;
            this.ch3_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch3_gain_ctrl.ValueChanged += new System.EventHandler(this.ch3_gain_ctrl_ValueChanged);
            // 
            // label50
            // 
            this.label50.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label50.Location = new System.Drawing.Point(283, 68);
            this.label50.Name = "label50";
            this.label50.Size = new System.Drawing.Size(77, 18);
            this.label50.TabIndex = 14;
            this.label50.Text = "Gain:";
            // 
            // DoF1box
            // 
            this.DoF1box.Controls.Add(this.pictureBox2);
            this.DoF1box.Controls.Add(this.Label41);
            this.DoF1box.Controls.Add(this.DoF1_mode_box);
            this.DoF1box.Controls.Add(this.pictureBox1);
            this.DoF1box.Controls.Add(this.Label56);
            this.DoF1box.Controls.Add(this.ch2_smax_label);
            this.DoF1box.Controls.Add(this.DoF1_flip_checkBox);
            this.DoF1box.Controls.Add(this.ch2_smin_label);
            this.DoF1box.Controls.Add(this.ch1_smax_label);
            this.DoF1box.Controls.Add(this.label25);
            this.DoF1box.Controls.Add(this.ch1_smin_label);
            this.DoF1box.Controls.Add(this.ch2_smin_tick);
            this.DoF1box.Controls.Add(this.ch2_smax_tick);
            this.DoF1box.Controls.Add(this.ch1_smin_tick);
            this.DoF1box.Controls.Add(this.ch1_smax_tick);
            this.DoF1box.Controls.Add(this.ch2_smax_ctrl);
            this.DoF1box.Controls.Add(this.ch1_smax_ctrl);
            this.DoF1box.Controls.Add(this.Label12);
            this.DoF1box.Controls.Add(this.ch2_smin_ctrl);
            this.DoF1box.Controls.Add(this.ch1_smin_ctrl);
            this.DoF1box.Controls.Add(this.label1);
            this.DoF1box.Controls.Add(this.Label8);
            this.DoF1box.Controls.Add(this.ch2_gain_ctrl);
            this.DoF1box.Controls.Add(this.label4);
            this.DoF1box.Controls.Add(this.label13);
            this.DoF1box.Controls.Add(this.MAV1_bar);
            this.DoF1box.Controls.Add(this.DoF1_mapping_combobox);
            this.DoF1box.Controls.Add(this.label14);
            this.DoF1box.Controls.Add(this.label15);
            this.DoF1box.Controls.Add(this.MAV2_bar);
            this.DoF1box.Controls.Add(this.ch1_gain_ctrl);
            this.DoF1box.Controls.Add(this.label16);
            this.DoF1box.Location = new System.Drawing.Point(11, 21);
            this.DoF1box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF1box.Name = "DoF1box";
            this.DoF1box.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF1box.Size = new System.Drawing.Size(451, 199);
            this.DoF1box.TabIndex = 4;
            this.DoF1box.TabStop = false;
            this.DoF1box.Text = "Degree of Freedom 1 - EMG (Forearm)";
            // 
            // pictureBox2
            // 
            this.pictureBox2.Image = global::brachIOplexus.Properties.Resources.bottom_arrow_rev2;
            this.pictureBox2.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox2.InitialImage")));
            this.pictureBox2.Location = new System.Drawing.Point(245, 137);
            this.pictureBox2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(29, 30);
            this.pictureBox2.TabIndex = 134;
            this.pictureBox2.TabStop = false;
            // 
            // Label41
            // 
            this.Label41.AutoSize = true;
            this.Label41.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label41.Location = new System.Drawing.Point(284, 30);
            this.Label41.Name = "Label41";
            this.Label41.Size = new System.Drawing.Size(17, 17);
            this.Label41.TabIndex = 127;
            this.Label41.Text = "&&";
            // 
            // DoF1_mode_box
            // 
            this.DoF1_mode_box.DisplayMember = "1";
            this.DoF1_mode_box.FormattingEnabled = true;
            this.DoF1_mode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.DoF1_mode_box.Location = new System.Drawing.Point(301, 27);
            this.DoF1_mode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF1_mode_box.Name = "DoF1_mode_box";
            this.DoF1_mode_box.Size = new System.Drawing.Size(105, 24);
            this.DoF1_mode_box.TabIndex = 130;
            this.DoF1_mode_box.SelectedIndexChanged += new System.EventHandler(this.DoF1_mode_box_SelectedIndexChanged);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = global::brachIOplexus.Properties.Resources.top_arrow_rev2;
            this.pictureBox1.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox1.InitialImage")));
            this.pictureBox1.Location = new System.Drawing.Point(245, 89);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(29, 30);
            this.pictureBox1.TabIndex = 133;
            this.pictureBox1.TabStop = false;
            // 
            // Label56
            // 
            this.Label56.AutoSize = true;
            this.Label56.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label56.Location = new System.Drawing.Point(83, 30);
            this.Label56.Name = "Label56";
            this.Label56.Size = new System.Drawing.Size(66, 17);
            this.Label56.TabIndex = 51;
            this.Label56.Text = "Ch1/Ch2 ";
            // 
            // ch2_smax_label
            // 
            this.ch2_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch2_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch2_smax_label.Location = new System.Drawing.Point(211, 172);
            this.ch2_smax_label.Name = "ch2_smax_label";
            this.ch2_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch2_smax_label.TabIndex = 45;
            this.ch2_smax_label.Text = "Smax";
            // 
            // DoF1_flip_checkBox
            // 
            this.DoF1_flip_checkBox.AutoSize = true;
            this.DoF1_flip_checkBox.Location = new System.Drawing.Point(259, 121);
            this.DoF1_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF1_flip_checkBox.Name = "DoF1_flip_checkBox";
            this.DoF1_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.DoF1_flip_checkBox.TabIndex = 132;
            this.DoF1_flip_checkBox.UseVisualStyleBackColor = true;
            this.DoF1_flip_checkBox.CheckedChanged += new System.EventHandler(this.DoF1_flip_checkBox_CheckedChanged);
            // 
            // ch2_smin_label
            // 
            this.ch2_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch2_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch2_smin_label.Location = new System.Drawing.Point(36, 172);
            this.ch2_smin_label.Name = "ch2_smin_label";
            this.ch2_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch2_smin_label.TabIndex = 44;
            this.ch2_smin_label.Text = "Smin";
            // 
            // ch1_smax_label
            // 
            this.ch1_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch1_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch1_smax_label.Location = new System.Drawing.Point(211, 119);
            this.ch1_smax_label.Name = "ch1_smax_label";
            this.ch1_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch1_smax_label.TabIndex = 43;
            this.ch1_smax_label.Text = "Smax";
            // 
            // label25
            // 
            this.label25.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label25.Location = new System.Drawing.Point(243, 68);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(35, 18);
            this.label25.TabIndex = 131;
            this.label25.Text = "Flip:";
            // 
            // ch1_smin_label
            // 
            this.ch1_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch1_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch1_smin_label.Location = new System.Drawing.Point(36, 119);
            this.ch1_smin_label.Name = "ch1_smin_label";
            this.ch1_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch1_smin_label.TabIndex = 42;
            this.ch1_smin_label.Text = "Smin";
            // 
            // ch2_smin_tick
            // 
            this.ch2_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch2_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch2_smin_tick.Location = new System.Drawing.Point(52, 143);
            this.ch2_smin_tick.Name = "ch2_smin_tick";
            this.ch2_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch2_smin_tick.TabIndex = 41;
            // 
            // ch2_smax_tick
            // 
            this.ch2_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch2_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch2_smax_tick.Location = new System.Drawing.Point(229, 143);
            this.ch2_smax_tick.Name = "ch2_smax_tick";
            this.ch2_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch2_smax_tick.TabIndex = 40;
            // 
            // ch1_smin_tick
            // 
            this.ch1_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch1_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch1_smin_tick.Location = new System.Drawing.Point(52, 90);
            this.ch1_smin_tick.Name = "ch1_smin_tick";
            this.ch1_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch1_smin_tick.TabIndex = 39;
            // 
            // ch1_smax_tick
            // 
            this.ch1_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch1_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch1_smax_tick.Location = new System.Drawing.Point(229, 90);
            this.ch1_smax_tick.Name = "ch1_smax_tick";
            this.ch1_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch1_smax_tick.TabIndex = 38;
            // 
            // ch2_smax_ctrl
            // 
            this.ch2_smax_ctrl.DecimalPlaces = 1;
            this.ch2_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch2_smax_ctrl.Location = new System.Drawing.Point(399, 145);
            this.ch2_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch2_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch2_smax_ctrl.Name = "ch2_smax_ctrl";
            this.ch2_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch2_smax_ctrl.TabIndex = 31;
            this.ch2_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch2_smax_ctrl.ValueChanged += new System.EventHandler(this.ch2_smax_ctrl_ValueChanged);
            // 
            // ch1_smax_ctrl
            // 
            this.ch1_smax_ctrl.DecimalPlaces = 1;
            this.ch1_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch1_smax_ctrl.Location = new System.Drawing.Point(399, 90);
            this.ch1_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch1_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch1_smax_ctrl.Name = "ch1_smax_ctrl";
            this.ch1_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch1_smax_ctrl.TabIndex = 30;
            this.ch1_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch1_smax_ctrl.ValueChanged += new System.EventHandler(this.ch1_smax_ctrl_ValueChanged);
            // 
            // Label12
            // 
            this.Label12.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label12.Location = new System.Drawing.Point(395, 68);
            this.Label12.Name = "Label12";
            this.Label12.Size = new System.Drawing.Size(47, 18);
            this.Label12.TabIndex = 29;
            this.Label12.Text = "Smax:";
            // 
            // ch2_smin_ctrl
            // 
            this.ch2_smin_ctrl.DecimalPlaces = 1;
            this.ch2_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch2_smin_ctrl.Location = new System.Drawing.Point(349, 145);
            this.ch2_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch2_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch2_smin_ctrl.Name = "ch2_smin_ctrl";
            this.ch2_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch2_smin_ctrl.TabIndex = 28;
            this.ch2_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch2_smin_ctrl.ValueChanged += new System.EventHandler(this.ch2_smin_ctrl_ValueChanged);
            // 
            // ch1_smin_ctrl
            // 
            this.ch1_smin_ctrl.DecimalPlaces = 1;
            this.ch1_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch1_smin_ctrl.Location = new System.Drawing.Point(349, 90);
            this.ch1_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch1_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch1_smin_ctrl.Name = "ch1_smin_ctrl";
            this.ch1_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch1_smin_ctrl.TabIndex = 27;
            this.ch1_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch1_smin_ctrl.ValueChanged += new System.EventHandler(this.ch1_smin_ctrl_ValueChanged);
            // 
            // label1
            // 
            this.label1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label1.Location = new System.Drawing.Point(347, 68);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(47, 18);
            this.label1.TabIndex = 26;
            this.label1.Text = "Smin:";
            // 
            // Label8
            // 
            this.Label8.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label8.Location = new System.Drawing.Point(49, 68);
            this.Label8.Name = "Label8";
            this.Label8.Size = new System.Drawing.Size(105, 18);
            this.Label8.TabIndex = 25;
            this.Label8.Text = "Signal Strength:";
            // 
            // ch2_gain_ctrl
            // 
            this.ch2_gain_ctrl.Location = new System.Drawing.Point(285, 145);
            this.ch2_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch2_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch2_gain_ctrl.Name = "ch2_gain_ctrl";
            this.ch2_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch2_gain_ctrl.TabIndex = 24;
            this.ch2_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch2_gain_ctrl.ValueChanged += new System.EventHandler(this.ch2_gain_ctrl_ValueChanged);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label4.Location = new System.Drawing.Point(13, 145);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(37, 17);
            this.label4.TabIndex = 23;
            this.label4.Text = "Ch2:";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label13.Location = new System.Drawing.Point(12, 90);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(37, 17);
            this.label13.TabIndex = 22;
            this.label13.Text = "Ch1:";
            // 
            // MAV1_bar
            // 
            this.MAV1_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV1_bar.Location = new System.Drawing.Point(52, 90);
            this.MAV1_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV1_bar.MarqueeAnimationSpeed = 30;
            this.MAV1_bar.Maximum = 500;
            this.MAV1_bar.Name = "MAV1_bar";
            this.MAV1_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV1_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV1_bar.TabIndex = 16;
            this.MAV1_bar.Value = 200;
            // 
            // DoF1_mapping_combobox
            // 
            this.DoF1_mapping_combobox.FormattingEnabled = true;
            this.DoF1_mapping_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.DoF1_mapping_combobox.Location = new System.Drawing.Point(175, 27);
            this.DoF1_mapping_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.DoF1_mapping_combobox.Name = "DoF1_mapping_combobox";
            this.DoF1_mapping_combobox.Size = new System.Drawing.Size(105, 24);
            this.DoF1_mapping_combobox.TabIndex = 21;
            this.DoF1_mapping_combobox.SelectedIndexChanged += new System.EventHandler(this.DoF1_mapping_combobox_SelectedIndexChanged);
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label14.Location = new System.Drawing.Point(147, 30);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(24, 17);
            this.label14.TabIndex = 20;
            this.label14.Text = ">>";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label15.Location = new System.Drawing.Point(13, 30);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(66, 17);
            this.label15.TabIndex = 18;
            this.label15.Text = "Mapping:";
            // 
            // MAV2_bar
            // 
            this.MAV2_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV2_bar.Location = new System.Drawing.Point(52, 143);
            this.MAV2_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV2_bar.MarqueeAnimationSpeed = 30;
            this.MAV2_bar.Maximum = 500;
            this.MAV2_bar.Name = "MAV2_bar";
            this.MAV2_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV2_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV2_bar.TabIndex = 17;
            this.MAV2_bar.Value = 200;
            // 
            // ch1_gain_ctrl
            // 
            this.ch1_gain_ctrl.Location = new System.Drawing.Point(285, 90);
            this.ch1_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch1_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch1_gain_ctrl.Name = "ch1_gain_ctrl";
            this.ch1_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch1_gain_ctrl.TabIndex = 15;
            this.ch1_gain_ctrl.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.ch1_gain_ctrl.ValueChanged += new System.EventHandler(this.ch1_gain_ctrl_ValueChanged);
            // 
            // label16
            // 
            this.label16.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label16.Location = new System.Drawing.Point(283, 68);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(77, 18);
            this.label16.TabIndex = 14;
            this.label16.Text = "Gain:";
            // 
            // SwitchBox
            // 
            this.SwitchBox.Controls.Add(this.label35);
            this.SwitchBox.Controls.Add(this.switch5_dofmode_box);
            this.SwitchBox.Controls.Add(this.label34);
            this.SwitchBox.Controls.Add(this.switch4_dofmode_box);
            this.SwitchBox.Controls.Add(this.label32);
            this.SwitchBox.Controls.Add(this.switch3_dofmode_box);
            this.SwitchBox.Controls.Add(this.label30);
            this.SwitchBox.Controls.Add(this.switch2_dofmode_box);
            this.SwitchBox.Controls.Add(this.label24);
            this.SwitchBox.Controls.Add(this.cctime_ctrl);
            this.SwitchBox.Controls.Add(this.switch1_dofmode_box);
            this.SwitchBox.Controls.Add(this.label2);
            this.SwitchBox.Controls.Add(this.text_checkBox);
            this.SwitchBox.Controls.Add(this.ch9_smax_label);
            this.SwitchBox.Controls.Add(this.ch9_smin_label);
            this.SwitchBox.Controls.Add(this.ch9_smin_tick);
            this.SwitchBox.Controls.Add(this.ch9_smax_tick);
            this.SwitchBox.Controls.Add(this.ch9_smax_ctrl);
            this.SwitchBox.Controls.Add(this.label97);
            this.SwitchBox.Controls.Add(this.ch9_smin_ctrl);
            this.SwitchBox.Controls.Add(this.label98);
            this.SwitchBox.Controls.Add(this.label99);
            this.SwitchBox.Controls.Add(this.label100);
            this.SwitchBox.Controls.Add(this.MAV9_bar);
            this.SwitchBox.Controls.Add(this.ch9_gain_ctrl);
            this.SwitchBox.Controls.Add(this.label101);
            this.SwitchBox.Controls.Add(this.led_checkBox);
            this.SwitchBox.Controls.Add(this.vocal_checkBox);
            this.SwitchBox.Controls.Add(this.ding_checkBox);
            this.SwitchBox.Controls.Add(this.label102);
            this.SwitchBox.Controls.Add(this.cycle5_flip_checkBox);
            this.SwitchBox.Controls.Add(this.cycle4_flip_checkBox);
            this.SwitchBox.Controls.Add(this.cycle3_flip_checkBox);
            this.SwitchBox.Controls.Add(this.cycle2_flip_checkBox);
            this.SwitchBox.Controls.Add(this.cycle1_flip_checkBox);
            this.SwitchBox.Controls.Add(this.label74);
            this.SwitchBox.Controls.Add(this.label70);
            this.SwitchBox.Controls.Add(this.label31);
            this.SwitchBox.Controls.Add(this.switch_mode_combobox);
            this.SwitchBox.Controls.Add(this.Switch_cycle5_combobox);
            this.SwitchBox.Controls.Add(this.Switch_cycle4_combobox);
            this.SwitchBox.Controls.Add(this.Switch_cycle3_combobox);
            this.SwitchBox.Controls.Add(this.Switch_cycle2_combobox);
            this.SwitchBox.Controls.Add(this.Switch_cycle1_combobox);
            this.SwitchBox.Controls.Add(this.cycle_number);
            this.SwitchBox.Controls.Add(this.label17);
            this.SwitchBox.Controls.Add(this.switch_dof_combobox);
            this.SwitchBox.Controls.Add(this.Label75);
            this.SwitchBox.Location = new System.Drawing.Point(923, 21);
            this.SwitchBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SwitchBox.Name = "SwitchBox";
            this.SwitchBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SwitchBox.Size = new System.Drawing.Size(469, 406);
            this.SwitchBox.TabIndex = 52;
            this.SwitchBox.TabStop = false;
            this.SwitchBox.Text = "Sequential Switch";
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label35.Location = new System.Drawing.Point(201, 268);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(17, 17);
            this.label35.TabIndex = 182;
            this.label35.Text = "&&";
            // 
            // switch5_dofmode_box
            // 
            this.switch5_dofmode_box.DisplayMember = "1";
            this.switch5_dofmode_box.FormattingEnabled = true;
            this.switch5_dofmode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.switch5_dofmode_box.Location = new System.Drawing.Point(219, 265);
            this.switch5_dofmode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch5_dofmode_box.Name = "switch5_dofmode_box";
            this.switch5_dofmode_box.Size = new System.Drawing.Size(105, 24);
            this.switch5_dofmode_box.TabIndex = 183;
            this.switch5_dofmode_box.SelectedIndexChanged += new System.EventHandler(this.switch5_dofmode_box_SelectedIndexChanged);
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label34.Location = new System.Drawing.Point(201, 244);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(17, 17);
            this.label34.TabIndex = 180;
            this.label34.Text = "&&";
            // 
            // switch4_dofmode_box
            // 
            this.switch4_dofmode_box.DisplayMember = "1";
            this.switch4_dofmode_box.FormattingEnabled = true;
            this.switch4_dofmode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.switch4_dofmode_box.Location = new System.Drawing.Point(219, 241);
            this.switch4_dofmode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch4_dofmode_box.Name = "switch4_dofmode_box";
            this.switch4_dofmode_box.Size = new System.Drawing.Size(105, 24);
            this.switch4_dofmode_box.TabIndex = 181;
            this.switch4_dofmode_box.SelectedIndexChanged += new System.EventHandler(this.switch4_dofmode_box_SelectedIndexChanged);
            // 
            // label32
            // 
            this.label32.AutoSize = true;
            this.label32.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label32.Location = new System.Drawing.Point(201, 220);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(17, 17);
            this.label32.TabIndex = 178;
            this.label32.Text = "&&";
            // 
            // switch3_dofmode_box
            // 
            this.switch3_dofmode_box.DisplayMember = "1";
            this.switch3_dofmode_box.FormattingEnabled = true;
            this.switch3_dofmode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.switch3_dofmode_box.Location = new System.Drawing.Point(219, 217);
            this.switch3_dofmode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch3_dofmode_box.Name = "switch3_dofmode_box";
            this.switch3_dofmode_box.Size = new System.Drawing.Size(105, 24);
            this.switch3_dofmode_box.TabIndex = 179;
            this.switch3_dofmode_box.SelectedIndexChanged += new System.EventHandler(this.switch3_dofmode_box_SelectedIndexChanged);
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label30.Location = new System.Drawing.Point(201, 196);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(17, 17);
            this.label30.TabIndex = 176;
            this.label30.Text = "&&";
            // 
            // switch2_dofmode_box
            // 
            this.switch2_dofmode_box.DisplayMember = "1";
            this.switch2_dofmode_box.FormattingEnabled = true;
            this.switch2_dofmode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.switch2_dofmode_box.Location = new System.Drawing.Point(219, 193);
            this.switch2_dofmode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch2_dofmode_box.Name = "switch2_dofmode_box";
            this.switch2_dofmode_box.Size = new System.Drawing.Size(105, 24);
            this.switch2_dofmode_box.TabIndex = 177;
            this.switch2_dofmode_box.SelectedIndexChanged += new System.EventHandler(this.switch2_dofmode_box_SelectedIndexChanged);
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label24.Location = new System.Drawing.Point(201, 172);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(17, 17);
            this.label24.TabIndex = 135;
            this.label24.Text = "&&";
            // 
            // cctime_ctrl
            // 
            this.cctime_ctrl.DecimalPlaces = 1;
            this.cctime_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.cctime_ctrl.Location = new System.Drawing.Point(412, 90);
            this.cctime_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cctime_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.cctime_ctrl.Name = "cctime_ctrl";
            this.cctime_ctrl.Size = new System.Drawing.Size(37, 22);
            this.cctime_ctrl.TabIndex = 175;
            this.cctime_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            65536});
            this.cctime_ctrl.ValueChanged += new System.EventHandler(this.cctime_ctrl_ValueChanged);
            // 
            // switch1_dofmode_box
            // 
            this.switch1_dofmode_box.DisplayMember = "1";
            this.switch1_dofmode_box.FormattingEnabled = true;
            this.switch1_dofmode_box.Items.AddRange(new object[] {
            "First to Smin",
            "Differential",
            "Single site"});
            this.switch1_dofmode_box.Location = new System.Drawing.Point(219, 169);
            this.switch1_dofmode_box.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch1_dofmode_box.Name = "switch1_dofmode_box";
            this.switch1_dofmode_box.Size = new System.Drawing.Size(105, 24);
            this.switch1_dofmode_box.TabIndex = 136;
            this.switch1_dofmode_box.SelectedIndexChanged += new System.EventHandler(this.switch1_dofmode_box_SelectedIndexChanged);
            // 
            // label2
            // 
            this.label2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label2.Location = new System.Drawing.Point(408, 68);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(55, 18);
            this.label2.TabIndex = 174;
            this.label2.Text = "CCtime:";
            // 
            // text_checkBox
            // 
            this.text_checkBox.AutoSize = true;
            this.text_checkBox.Location = new System.Drawing.Point(337, 327);
            this.text_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.text_checkBox.Name = "text_checkBox";
            this.text_checkBox.Size = new System.Drawing.Size(57, 21);
            this.text_checkBox.TabIndex = 173;
            this.text_checkBox.Text = "Text";
            this.text_checkBox.UseVisualStyleBackColor = true;
            // 
            // ch9_smax_label
            // 
            this.ch9_smax_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch9_smax_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch9_smax_label.Location = new System.Drawing.Point(211, 119);
            this.ch9_smax_label.Name = "ch9_smax_label";
            this.ch9_smax_label.Size = new System.Drawing.Size(43, 18);
            this.ch9_smax_label.TabIndex = 172;
            this.ch9_smax_label.Text = "Smax";
            // 
            // ch9_smin_label
            // 
            this.ch9_smin_label.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.ch9_smin_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch9_smin_label.Location = new System.Drawing.Point(36, 119);
            this.ch9_smin_label.Name = "ch9_smin_label";
            this.ch9_smin_label.Size = new System.Drawing.Size(41, 18);
            this.ch9_smin_label.TabIndex = 171;
            this.ch9_smin_label.Text = "Smin";
            // 
            // ch9_smin_tick
            // 
            this.ch9_smin_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch9_smin_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch9_smin_tick.Location = new System.Drawing.Point(52, 90);
            this.ch9_smin_tick.Name = "ch9_smin_tick";
            this.ch9_smin_tick.Size = new System.Drawing.Size(3, 30);
            this.ch9_smin_tick.TabIndex = 170;
            // 
            // ch9_smax_tick
            // 
            this.ch9_smax_tick.BackColor = System.Drawing.Color.MediumPurple;
            this.ch9_smax_tick.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ch9_smax_tick.Location = new System.Drawing.Point(229, 90);
            this.ch9_smax_tick.Name = "ch9_smax_tick";
            this.ch9_smax_tick.Size = new System.Drawing.Size(3, 30);
            this.ch9_smax_tick.TabIndex = 169;
            // 
            // ch9_smax_ctrl
            // 
            this.ch9_smax_ctrl.DecimalPlaces = 1;
            this.ch9_smax_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch9_smax_ctrl.Location = new System.Drawing.Point(361, 90);
            this.ch9_smax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch9_smax_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch9_smax_ctrl.Name = "ch9_smax_ctrl";
            this.ch9_smax_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch9_smax_ctrl.TabIndex = 168;
            this.ch9_smax_ctrl.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch9_smax_ctrl.ValueChanged += new System.EventHandler(this.ch9_smax_ctrl_ValueChanged);
            // 
            // label97
            // 
            this.label97.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label97.Location = new System.Drawing.Point(357, 68);
            this.label97.Name = "label97";
            this.label97.Size = new System.Drawing.Size(47, 18);
            this.label97.TabIndex = 167;
            this.label97.Text = "Smax:";
            // 
            // ch9_smin_ctrl
            // 
            this.ch9_smin_ctrl.DecimalPlaces = 1;
            this.ch9_smin_ctrl.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.ch9_smin_ctrl.Location = new System.Drawing.Point(311, 90);
            this.ch9_smin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch9_smin_ctrl.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.ch9_smin_ctrl.Name = "ch9_smin_ctrl";
            this.ch9_smin_ctrl.Size = new System.Drawing.Size(37, 22);
            this.ch9_smin_ctrl.TabIndex = 166;
            this.ch9_smin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch9_smin_ctrl.ValueChanged += new System.EventHandler(this.ch9_smin_ctrl_ValueChanged);
            // 
            // label98
            // 
            this.label98.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label98.Location = new System.Drawing.Point(308, 68);
            this.label98.Name = "label98";
            this.label98.Size = new System.Drawing.Size(47, 18);
            this.label98.TabIndex = 165;
            this.label98.Text = "Smin:";
            // 
            // label99
            // 
            this.label99.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label99.Location = new System.Drawing.Point(49, 68);
            this.label99.Name = "label99";
            this.label99.Size = new System.Drawing.Size(105, 18);
            this.label99.TabIndex = 164;
            this.label99.Text = "Signal Strength:";
            // 
            // label100
            // 
            this.label100.AutoSize = true;
            this.label100.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label100.Location = new System.Drawing.Point(12, 90);
            this.label100.Name = "label100";
            this.label100.Size = new System.Drawing.Size(37, 17);
            this.label100.TabIndex = 163;
            this.label100.Text = "Ch9:";
            // 
            // MAV9_bar
            // 
            this.MAV9_bar.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MAV9_bar.Location = new System.Drawing.Point(52, 90);
            this.MAV9_bar.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MAV9_bar.MarqueeAnimationSpeed = 30;
            this.MAV9_bar.Maximum = 500;
            this.MAV9_bar.Name = "MAV9_bar";
            this.MAV9_bar.Size = new System.Drawing.Size(179, 27);
            this.MAV9_bar.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.MAV9_bar.TabIndex = 162;
            this.MAV9_bar.Value = 200;
            // 
            // ch9_gain_ctrl
            // 
            this.ch9_gain_ctrl.Location = new System.Drawing.Point(248, 90);
            this.ch9_gain_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ch9_gain_ctrl.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.ch9_gain_ctrl.Name = "ch9_gain_ctrl";
            this.ch9_gain_ctrl.Size = new System.Drawing.Size(51, 22);
            this.ch9_gain_ctrl.TabIndex = 161;
            this.ch9_gain_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.ch9_gain_ctrl.ValueChanged += new System.EventHandler(this.ch9_gain_ctrl_ValueChanged);
            // 
            // label101
            // 
            this.label101.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label101.Location = new System.Drawing.Point(245, 68);
            this.label101.Name = "label101";
            this.label101.Size = new System.Drawing.Size(77, 18);
            this.label101.TabIndex = 160;
            this.label101.Text = "Gain:";
            // 
            // led_checkBox
            // 
            this.led_checkBox.AutoSize = true;
            this.led_checkBox.Enabled = false;
            this.led_checkBox.Location = new System.Drawing.Point(400, 353);
            this.led_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.led_checkBox.Name = "led_checkBox";
            this.led_checkBox.Size = new System.Drawing.Size(57, 21);
            this.led_checkBox.TabIndex = 159;
            this.led_checkBox.Text = "LED";
            this.led_checkBox.UseVisualStyleBackColor = true;
            // 
            // vocal_checkBox
            // 
            this.vocal_checkBox.AutoSize = true;
            this.vocal_checkBox.Location = new System.Drawing.Point(337, 353);
            this.vocal_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.vocal_checkBox.Name = "vocal_checkBox";
            this.vocal_checkBox.Size = new System.Drawing.Size(65, 21);
            this.vocal_checkBox.TabIndex = 158;
            this.vocal_checkBox.Text = "Vocal";
            this.vocal_checkBox.UseVisualStyleBackColor = true;
            // 
            // ding_checkBox
            // 
            this.ding_checkBox.AutoSize = true;
            this.ding_checkBox.Location = new System.Drawing.Point(400, 327);
            this.ding_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ding_checkBox.Name = "ding_checkBox";
            this.ding_checkBox.Size = new System.Drawing.Size(59, 21);
            this.ding_checkBox.TabIndex = 157;
            this.ding_checkBox.Text = "Ding";
            this.ding_checkBox.UseVisualStyleBackColor = true;
            // 
            // label102
            // 
            this.label102.AutoSize = true;
            this.label102.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label102.Location = new System.Drawing.Point(333, 308);
            this.label102.Name = "label102";
            this.label102.Size = new System.Drawing.Size(74, 17);
            this.label102.TabIndex = 156;
            this.label102.Text = "Feedback:";
            // 
            // cycle5_flip_checkBox
            // 
            this.cycle5_flip_checkBox.AutoSize = true;
            this.cycle5_flip_checkBox.Location = new System.Drawing.Point(345, 271);
            this.cycle5_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cycle5_flip_checkBox.Name = "cycle5_flip_checkBox";
            this.cycle5_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.cycle5_flip_checkBox.TabIndex = 155;
            this.cycle5_flip_checkBox.UseVisualStyleBackColor = true;
            this.cycle5_flip_checkBox.CheckedChanged += new System.EventHandler(this.cycle5_flip_checkBox_CheckedChanged);
            // 
            // cycle4_flip_checkBox
            // 
            this.cycle4_flip_checkBox.AutoSize = true;
            this.cycle4_flip_checkBox.Location = new System.Drawing.Point(345, 247);
            this.cycle4_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cycle4_flip_checkBox.Name = "cycle4_flip_checkBox";
            this.cycle4_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.cycle4_flip_checkBox.TabIndex = 154;
            this.cycle4_flip_checkBox.UseVisualStyleBackColor = true;
            this.cycle4_flip_checkBox.CheckedChanged += new System.EventHandler(this.cycle4_flip_checkBox_CheckedChanged);
            // 
            // cycle3_flip_checkBox
            // 
            this.cycle3_flip_checkBox.AutoSize = true;
            this.cycle3_flip_checkBox.Location = new System.Drawing.Point(345, 223);
            this.cycle3_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cycle3_flip_checkBox.Name = "cycle3_flip_checkBox";
            this.cycle3_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.cycle3_flip_checkBox.TabIndex = 153;
            this.cycle3_flip_checkBox.UseVisualStyleBackColor = true;
            this.cycle3_flip_checkBox.CheckedChanged += new System.EventHandler(this.cycle3_flip_checkBox_CheckedChanged);
            // 
            // cycle2_flip_checkBox
            // 
            this.cycle2_flip_checkBox.AutoSize = true;
            this.cycle2_flip_checkBox.Location = new System.Drawing.Point(345, 199);
            this.cycle2_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cycle2_flip_checkBox.Name = "cycle2_flip_checkBox";
            this.cycle2_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.cycle2_flip_checkBox.TabIndex = 152;
            this.cycle2_flip_checkBox.UseVisualStyleBackColor = true;
            this.cycle2_flip_checkBox.CheckedChanged += new System.EventHandler(this.cycle2_flip_checkBox_CheckedChanged);
            // 
            // cycle1_flip_checkBox
            // 
            this.cycle1_flip_checkBox.AutoSize = true;
            this.cycle1_flip_checkBox.Location = new System.Drawing.Point(345, 175);
            this.cycle1_flip_checkBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cycle1_flip_checkBox.Name = "cycle1_flip_checkBox";
            this.cycle1_flip_checkBox.Size = new System.Drawing.Size(18, 17);
            this.cycle1_flip_checkBox.TabIndex = 151;
            this.cycle1_flip_checkBox.UseVisualStyleBackColor = true;
            this.cycle1_flip_checkBox.CheckedChanged += new System.EventHandler(this.cycle1_flip_checkBox_CheckedChanged);
            // 
            // label74
            // 
            this.label74.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label74.Location = new System.Drawing.Point(333, 153);
            this.label74.Name = "label74";
            this.label74.Size = new System.Drawing.Size(35, 18);
            this.label74.TabIndex = 150;
            this.label74.Text = "Flip:";
            // 
            // label70
            // 
            this.label70.AutoSize = true;
            this.label70.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label70.Location = new System.Drawing.Point(11, 172);
            this.label70.Name = "label70";
            this.label70.Size = new System.Drawing.Size(78, 17);
            this.label70.TabIndex = 149;
            this.label70.Text = "Switch List:";
            // 
            // label31
            // 
            this.label31.AutoSize = true;
            this.label31.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label31.Location = new System.Drawing.Point(197, 30);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(17, 17);
            this.label31.TabIndex = 147;
            this.label31.Text = "&&";
            // 
            // switch_mode_combobox
            // 
            this.switch_mode_combobox.DisplayMember = "1";
            this.switch_mode_combobox.FormattingEnabled = true;
            this.switch_mode_combobox.Items.AddRange(new object[] {
            "EMG Ch9",
            "Joystick Click",
            "Co-contraction",
            "Adaptive"});
            this.switch_mode_combobox.Location = new System.Drawing.Point(215, 27);
            this.switch_mode_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch_mode_combobox.Name = "switch_mode_combobox";
            this.switch_mode_combobox.Size = new System.Drawing.Size(120, 24);
            this.switch_mode_combobox.TabIndex = 148;
            this.switch_mode_combobox.SelectedIndexChanged += new System.EventHandler(this.switch_mode_combobox_SelectedIndexChanged);
            // 
            // Switch_cycle5_combobox
            // 
            this.Switch_cycle5_combobox.FormattingEnabled = true;
            this.Switch_cycle5_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.Switch_cycle5_combobox.Location = new System.Drawing.Point(92, 265);
            this.Switch_cycle5_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Switch_cycle5_combobox.Name = "Switch_cycle5_combobox";
            this.Switch_cycle5_combobox.Size = new System.Drawing.Size(105, 24);
            this.Switch_cycle5_combobox.TabIndex = 73;
            this.Switch_cycle5_combobox.SelectedIndexChanged += new System.EventHandler(this.Switch_cycle5_combobox_SelectedIndexChanged);
            // 
            // Switch_cycle4_combobox
            // 
            this.Switch_cycle4_combobox.FormattingEnabled = true;
            this.Switch_cycle4_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.Switch_cycle4_combobox.Location = new System.Drawing.Point(92, 241);
            this.Switch_cycle4_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Switch_cycle4_combobox.Name = "Switch_cycle4_combobox";
            this.Switch_cycle4_combobox.Size = new System.Drawing.Size(105, 24);
            this.Switch_cycle4_combobox.TabIndex = 72;
            this.Switch_cycle4_combobox.SelectedIndexChanged += new System.EventHandler(this.Switch_cycle4_combobox_SelectedIndexChanged);
            // 
            // Switch_cycle3_combobox
            // 
            this.Switch_cycle3_combobox.FormattingEnabled = true;
            this.Switch_cycle3_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.Switch_cycle3_combobox.Location = new System.Drawing.Point(92, 217);
            this.Switch_cycle3_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Switch_cycle3_combobox.Name = "Switch_cycle3_combobox";
            this.Switch_cycle3_combobox.Size = new System.Drawing.Size(105, 24);
            this.Switch_cycle3_combobox.TabIndex = 71;
            this.Switch_cycle3_combobox.SelectedIndexChanged += new System.EventHandler(this.Switch_cycle3_combobox_SelectedIndexChanged);
            // 
            // Switch_cycle2_combobox
            // 
            this.Switch_cycle2_combobox.FormattingEnabled = true;
            this.Switch_cycle2_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.Switch_cycle2_combobox.Location = new System.Drawing.Point(92, 193);
            this.Switch_cycle2_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Switch_cycle2_combobox.Name = "Switch_cycle2_combobox";
            this.Switch_cycle2_combobox.Size = new System.Drawing.Size(105, 24);
            this.Switch_cycle2_combobox.TabIndex = 70;
            this.Switch_cycle2_combobox.SelectedIndexChanged += new System.EventHandler(this.Switch_cycle2_combobox_SelectedIndexChanged);
            // 
            // Switch_cycle1_combobox
            // 
            this.Switch_cycle1_combobox.FormattingEnabled = true;
            this.Switch_cycle1_combobox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.Switch_cycle1_combobox.Location = new System.Drawing.Point(92, 169);
            this.Switch_cycle1_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Switch_cycle1_combobox.Name = "Switch_cycle1_combobox";
            this.Switch_cycle1_combobox.Size = new System.Drawing.Size(105, 24);
            this.Switch_cycle1_combobox.TabIndex = 53;
            this.Switch_cycle1_combobox.SelectedIndexChanged += new System.EventHandler(this.Switch_cycle1_combobox_SelectedIndexChanged);
            // 
            // cycle_number
            // 
            this.cycle_number.AutoSize = true;
            this.cycle_number.Font = new System.Drawing.Font("Microsoft Sans Serif", 28.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cycle_number.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.cycle_number.Location = new System.Drawing.Point(12, 327);
            this.cycle_number.Name = "cycle_number";
            this.cycle_number.Size = new System.Drawing.Size(56, 55);
            this.cycle_number.TabIndex = 69;
            this.cycle_number.Text = "--";
            this.cycle_number.TextChanged += new System.EventHandler(this.cycle_number_TextChanged);
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label17.Location = new System.Drawing.Point(12, 311);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(84, 17);
            this.label17.TabIndex = 68;
            this.label17.Text = "Switched to:";
            // 
            // switch_dof_combobox
            // 
            this.switch_dof_combobox.FormattingEnabled = true;
            this.switch_dof_combobox.Items.AddRange(new object[] {
            "Off",
            "Ch1/Ch2",
            "Ch3/Ch4",
            "Ch5/Ch6",
            "Ch7/Ch8"});
            this.switch_dof_combobox.Location = new System.Drawing.Point(92, 27);
            this.switch_dof_combobox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch_dof_combobox.Name = "switch_dof_combobox";
            this.switch_dof_combobox.Size = new System.Drawing.Size(103, 24);
            this.switch_dof_combobox.TabIndex = 19;
            this.switch_dof_combobox.SelectedIndexChanged += new System.EventHandler(this.switch_dof_combobox_SelectedIndexChanged);
            // 
            // Label75
            // 
            this.Label75.AutoSize = true;
            this.Label75.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label75.Location = new System.Drawing.Point(12, 30);
            this.Label75.Name = "Label75";
            this.Label75.Size = new System.Drawing.Size(66, 17);
            this.Label75.TabIndex = 18;
            this.Label75.Text = "Mapping:";
            // 
            // RobotBox
            // 
            this.RobotBox.Controls.Add(this.arm_label);
            this.RobotBox.Controls.Add(this.RAM_text);
            this.RobotBox.Controls.Add(this.label110);
            this.RobotBox.Controls.Add(this.label29);
            this.RobotBox.Controls.Add(this.AX12stopBTN);
            this.RobotBox.Controls.Add(this.AX12startBTN);
            this.RobotBox.Controls.Add(this.hand_comboBox);
            this.RobotBox.Controls.Add(this.label23);
            this.RobotBox.Location = new System.Drawing.Point(13, 513);
            this.RobotBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotBox.Name = "RobotBox";
            this.RobotBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotBox.Size = new System.Drawing.Size(835, 270);
            this.RobotBox.TabIndex = 137;
            this.RobotBox.TabStop = false;
            this.RobotBox.Text = "Robotic Arm";
            // 
            // arm_label
            // 
            this.arm_label.BackColor = System.Drawing.Color.White;
            this.arm_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.arm_label.Location = new System.Drawing.Point(307, 32);
            this.arm_label.Name = "arm_label";
            this.arm_label.Size = new System.Drawing.Size(141, 18);
            this.arm_label.TabIndex = 143;
            this.arm_label.Text = "Bento Arm";
            // 
            // RAM_text
            // 
            this.RAM_text.AutoSize = true;
            this.RAM_text.Font = new System.Drawing.Font("Microsoft Sans Serif", 28.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.RAM_text.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.RAM_text.Location = new System.Drawing.Point(775, 12);
            this.RAM_text.Name = "RAM_text";
            this.RAM_text.Size = new System.Drawing.Size(56, 55);
            this.RAM_text.TabIndex = 141;
            this.RAM_text.Text = "--";
            // 
            // label110
            // 
            this.label110.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label110.Location = new System.Drawing.Point(233, 32);
            this.label110.Name = "label110";
            this.label110.Size = new System.Drawing.Size(79, 18);
            this.label110.TabIndex = 142;
            this.label110.Text = "Arm Type:";
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label29.Location = new System.Drawing.Point(684, 32);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(90, 17);
            this.label29.TabIndex = 140;
            this.label29.Text = "RAM (bytes):";
            // 
            // AX12stopBTN
            // 
            this.AX12stopBTN.Enabled = false;
            this.AX12stopBTN.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.AX12stopBTN.Location = new System.Drawing.Point(101, 26);
            this.AX12stopBTN.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.AX12stopBTN.Name = "AX12stopBTN";
            this.AX12stopBTN.Size = new System.Drawing.Size(83, 28);
            this.AX12stopBTN.TabIndex = 140;
            this.AX12stopBTN.Text = "Stop";
            this.AX12stopBTN.Click += new System.EventHandler(this.AX12stopBTN_Click);
            // 
            // AX12startBTN
            // 
            this.AX12startBTN.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.AX12startBTN.Location = new System.Drawing.Point(11, 26);
            this.AX12startBTN.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.AX12startBTN.Name = "AX12startBTN";
            this.AX12startBTN.Size = new System.Drawing.Size(83, 28);
            this.AX12startBTN.TabIndex = 139;
            this.AX12startBTN.Text = "Start";
            this.AX12startBTN.Click += new System.EventHandler(this.AX12startBTN_Click);
            // 
            // hand_comboBox
            // 
            this.hand_comboBox.FormattingEnabled = true;
            this.hand_comboBox.Items.AddRange(new object[] {
            "AX18 (Single)",
            "AX18 (Dual)",
            "MX28 (Single)",
            "MX28 (Dual)",
            "Handi-Hand",
            "Commercial Myo"});
            this.hand_comboBox.Location = new System.Drawing.Point(545, 30);
            this.hand_comboBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_comboBox.Name = "hand_comboBox";
            this.hand_comboBox.Size = new System.Drawing.Size(133, 24);
            this.hand_comboBox.TabIndex = 93;
            // 
            // label23
            // 
            this.label23.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label23.Location = new System.Drawing.Point(457, 32);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(85, 18);
            this.label23.TabIndex = 92;
            this.label23.Text = "Hand Type:";
            // 
            // RobotFeedbackBox
            // 
            this.RobotFeedbackBox.Controls.Add(this.Temp5);
            this.RobotFeedbackBox.Controls.Add(this.Volt5);
            this.RobotFeedbackBox.Controls.Add(this.Load5);
            this.RobotFeedbackBox.Controls.Add(this.Vel5);
            this.RobotFeedbackBox.Controls.Add(this.Pos5);
            this.RobotFeedbackBox.Controls.Add(this.Temp3);
            this.RobotFeedbackBox.Controls.Add(this.Volt3);
            this.RobotFeedbackBox.Controls.Add(this.Load3);
            this.RobotFeedbackBox.Controls.Add(this.Vel3);
            this.RobotFeedbackBox.Controls.Add(this.Pos3);
            this.RobotFeedbackBox.Controls.Add(this.Temp2);
            this.RobotFeedbackBox.Controls.Add(this.Volt2);
            this.RobotFeedbackBox.Controls.Add(this.Load2);
            this.RobotFeedbackBox.Controls.Add(this.Vel2);
            this.RobotFeedbackBox.Controls.Add(this.Pos2);
            this.RobotFeedbackBox.Controls.Add(this.Temp1);
            this.RobotFeedbackBox.Controls.Add(this.Volt1);
            this.RobotFeedbackBox.Controls.Add(this.Load1);
            this.RobotFeedbackBox.Controls.Add(this.Vel1);
            this.RobotFeedbackBox.Controls.Add(this.Pos1);
            this.RobotFeedbackBox.Controls.Add(this.label109);
            this.RobotFeedbackBox.Controls.Add(this.label108);
            this.RobotFeedbackBox.Controls.Add(this.label107);
            this.RobotFeedbackBox.Controls.Add(this.Temp4);
            this.RobotFeedbackBox.Controls.Add(this.label106);
            this.RobotFeedbackBox.Controls.Add(this.label200);
            this.RobotFeedbackBox.Controls.Add(this.Volt4);
            this.RobotFeedbackBox.Controls.Add(this.Load4);
            this.RobotFeedbackBox.Controls.Add(this.Vel4);
            this.RobotFeedbackBox.Controls.Add(this.Pos4);
            this.RobotFeedbackBox.Enabled = false;
            this.RobotFeedbackBox.Location = new System.Drawing.Point(501, 161);
            this.RobotFeedbackBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotFeedbackBox.Name = "RobotFeedbackBox";
            this.RobotFeedbackBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotFeedbackBox.Size = new System.Drawing.Size(303, 187);
            this.RobotFeedbackBox.TabIndex = 141;
            this.RobotFeedbackBox.TabStop = false;
            this.RobotFeedbackBox.Text = "Feedback";
            // 
            // Temp5
            // 
            this.Temp5.AutoSize = true;
            this.Temp5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Temp5.Location = new System.Drawing.Point(245, 156);
            this.Temp5.Name = "Temp5";
            this.Temp5.Size = new System.Drawing.Size(18, 17);
            this.Temp5.TabIndex = 169;
            this.Temp5.Text = "--";
            // 
            // Volt5
            // 
            this.Volt5.AutoSize = true;
            this.Volt5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Volt5.Location = new System.Drawing.Point(200, 156);
            this.Volt5.Name = "Volt5";
            this.Volt5.Size = new System.Drawing.Size(18, 17);
            this.Volt5.TabIndex = 168;
            this.Volt5.Text = "--";
            // 
            // Load5
            // 
            this.Load5.AutoSize = true;
            this.Load5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Load5.Location = new System.Drawing.Point(147, 156);
            this.Load5.Name = "Load5";
            this.Load5.Size = new System.Drawing.Size(18, 17);
            this.Load5.TabIndex = 167;
            this.Load5.Text = "--";
            // 
            // Vel5
            // 
            this.Vel5.AutoSize = true;
            this.Vel5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Vel5.Location = new System.Drawing.Point(77, 156);
            this.Vel5.Name = "Vel5";
            this.Vel5.Size = new System.Drawing.Size(18, 17);
            this.Vel5.TabIndex = 166;
            this.Vel5.Text = "--";
            // 
            // Pos5
            // 
            this.Pos5.AutoSize = true;
            this.Pos5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Pos5.Location = new System.Drawing.Point(7, 156);
            this.Pos5.Name = "Pos5";
            this.Pos5.Size = new System.Drawing.Size(18, 17);
            this.Pos5.TabIndex = 165;
            this.Pos5.Text = "--";
            // 
            // Temp3
            // 
            this.Temp3.AutoSize = true;
            this.Temp3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Temp3.Location = new System.Drawing.Point(245, 97);
            this.Temp3.Name = "Temp3";
            this.Temp3.Size = new System.Drawing.Size(18, 17);
            this.Temp3.TabIndex = 163;
            this.Temp3.Text = "--";
            // 
            // Volt3
            // 
            this.Volt3.AutoSize = true;
            this.Volt3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Volt3.Location = new System.Drawing.Point(200, 97);
            this.Volt3.Name = "Volt3";
            this.Volt3.Size = new System.Drawing.Size(18, 17);
            this.Volt3.TabIndex = 162;
            this.Volt3.Text = "--";
            // 
            // Load3
            // 
            this.Load3.AutoSize = true;
            this.Load3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Load3.Location = new System.Drawing.Point(147, 97);
            this.Load3.Name = "Load3";
            this.Load3.Size = new System.Drawing.Size(18, 17);
            this.Load3.TabIndex = 161;
            this.Load3.Text = "--";
            // 
            // Vel3
            // 
            this.Vel3.AutoSize = true;
            this.Vel3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Vel3.Location = new System.Drawing.Point(77, 97);
            this.Vel3.Name = "Vel3";
            this.Vel3.Size = new System.Drawing.Size(18, 17);
            this.Vel3.TabIndex = 160;
            this.Vel3.Text = "--";
            // 
            // Pos3
            // 
            this.Pos3.AutoSize = true;
            this.Pos3.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Pos3.Location = new System.Drawing.Point(7, 97);
            this.Pos3.Name = "Pos3";
            this.Pos3.Size = new System.Drawing.Size(18, 17);
            this.Pos3.TabIndex = 159;
            this.Pos3.Text = "--";
            // 
            // Temp2
            // 
            this.Temp2.AutoSize = true;
            this.Temp2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Temp2.Location = new System.Drawing.Point(245, 68);
            this.Temp2.Name = "Temp2";
            this.Temp2.Size = new System.Drawing.Size(18, 17);
            this.Temp2.TabIndex = 157;
            this.Temp2.Text = "--";
            // 
            // Volt2
            // 
            this.Volt2.AutoSize = true;
            this.Volt2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Volt2.Location = new System.Drawing.Point(199, 68);
            this.Volt2.Name = "Volt2";
            this.Volt2.Size = new System.Drawing.Size(18, 17);
            this.Volt2.TabIndex = 156;
            this.Volt2.Text = "--";
            // 
            // Load2
            // 
            this.Load2.AutoSize = true;
            this.Load2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Load2.Location = new System.Drawing.Point(147, 68);
            this.Load2.Name = "Load2";
            this.Load2.Size = new System.Drawing.Size(18, 17);
            this.Load2.TabIndex = 155;
            this.Load2.Text = "--";
            // 
            // Vel2
            // 
            this.Vel2.AutoSize = true;
            this.Vel2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Vel2.Location = new System.Drawing.Point(76, 68);
            this.Vel2.Name = "Vel2";
            this.Vel2.Size = new System.Drawing.Size(18, 17);
            this.Vel2.TabIndex = 154;
            this.Vel2.Text = "--";
            // 
            // Pos2
            // 
            this.Pos2.AutoSize = true;
            this.Pos2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Pos2.Location = new System.Drawing.Point(5, 68);
            this.Pos2.Name = "Pos2";
            this.Pos2.Size = new System.Drawing.Size(18, 17);
            this.Pos2.TabIndex = 153;
            this.Pos2.Text = "--";
            // 
            // Temp1
            // 
            this.Temp1.AutoSize = true;
            this.Temp1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Temp1.Location = new System.Drawing.Point(245, 39);
            this.Temp1.Name = "Temp1";
            this.Temp1.Size = new System.Drawing.Size(18, 17);
            this.Temp1.TabIndex = 151;
            this.Temp1.Text = "--";
            // 
            // Volt1
            // 
            this.Volt1.AutoSize = true;
            this.Volt1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Volt1.Location = new System.Drawing.Point(199, 39);
            this.Volt1.Name = "Volt1";
            this.Volt1.Size = new System.Drawing.Size(18, 17);
            this.Volt1.TabIndex = 150;
            this.Volt1.Text = "--";
            // 
            // Load1
            // 
            this.Load1.AutoSize = true;
            this.Load1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Load1.Location = new System.Drawing.Point(147, 39);
            this.Load1.Name = "Load1";
            this.Load1.Size = new System.Drawing.Size(18, 17);
            this.Load1.TabIndex = 149;
            this.Load1.Text = "--";
            // 
            // Vel1
            // 
            this.Vel1.AutoSize = true;
            this.Vel1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Vel1.Location = new System.Drawing.Point(76, 39);
            this.Vel1.Name = "Vel1";
            this.Vel1.Size = new System.Drawing.Size(18, 17);
            this.Vel1.TabIndex = 148;
            this.Vel1.Text = "--";
            // 
            // Pos1
            // 
            this.Pos1.AutoSize = true;
            this.Pos1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Pos1.Location = new System.Drawing.Point(5, 39);
            this.Pos1.Name = "Pos1";
            this.Pos1.Size = new System.Drawing.Size(18, 17);
            this.Pos1.TabIndex = 147;
            this.Pos1.Text = "--";
            // 
            // label109
            // 
            this.label109.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label109.Location = new System.Drawing.Point(245, 15);
            this.label109.Name = "label109";
            this.label109.Size = new System.Drawing.Size(51, 25);
            this.label109.TabIndex = 145;
            this.label109.Text = "Temp:";
            // 
            // label108
            // 
            this.label108.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label108.Location = new System.Drawing.Point(199, 15);
            this.label108.Name = "label108";
            this.label108.Size = new System.Drawing.Size(41, 25);
            this.label108.TabIndex = 144;
            this.label108.Text = "Volt:";
            // 
            // label107
            // 
            this.label107.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label107.Location = new System.Drawing.Point(147, 15);
            this.label107.Name = "label107";
            this.label107.Size = new System.Drawing.Size(52, 25);
            this.label107.TabIndex = 143;
            this.label107.Text = "Load:";
            // 
            // Temp4
            // 
            this.Temp4.AutoSize = true;
            this.Temp4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Temp4.Location = new System.Drawing.Point(245, 127);
            this.Temp4.Name = "Temp4";
            this.Temp4.Size = new System.Drawing.Size(18, 17);
            this.Temp4.TabIndex = 137;
            this.Temp4.Text = "--";
            // 
            // label106
            // 
            this.label106.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label106.Location = new System.Drawing.Point(76, 15);
            this.label106.Name = "label106";
            this.label106.Size = new System.Drawing.Size(65, 25);
            this.label106.TabIndex = 142;
            this.label106.Text = "Velocity:";
            // 
            // label200
            // 
            this.label200.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label200.Location = new System.Drawing.Point(5, 15);
            this.label200.Name = "label200";
            this.label200.Size = new System.Drawing.Size(65, 25);
            this.label200.TabIndex = 140;
            this.label200.Text = "Position:";
            // 
            // Volt4
            // 
            this.Volt4.AutoSize = true;
            this.Volt4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Volt4.Location = new System.Drawing.Point(200, 127);
            this.Volt4.Name = "Volt4";
            this.Volt4.Size = new System.Drawing.Size(18, 17);
            this.Volt4.TabIndex = 132;
            this.Volt4.Text = "--";
            // 
            // Load4
            // 
            this.Load4.AutoSize = true;
            this.Load4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Load4.Location = new System.Drawing.Point(147, 127);
            this.Load4.Name = "Load4";
            this.Load4.Size = new System.Drawing.Size(18, 17);
            this.Load4.TabIndex = 127;
            this.Load4.Text = "--";
            // 
            // Vel4
            // 
            this.Vel4.AutoSize = true;
            this.Vel4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Vel4.Location = new System.Drawing.Point(77, 127);
            this.Vel4.Name = "Vel4";
            this.Vel4.Size = new System.Drawing.Size(18, 17);
            this.Vel4.TabIndex = 122;
            this.Vel4.Text = "--";
            // 
            // Pos4
            // 
            this.Pos4.AutoSize = true;
            this.Pos4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Pos4.Location = new System.Drawing.Point(7, 127);
            this.Pos4.Name = "Pos4";
            this.Pos4.Size = new System.Drawing.Size(18, 17);
            this.Pos4.TabIndex = 117;
            this.Pos4.Text = "--";
            // 
            // RobotParamBox
            // 
            this.RobotParamBox.Controls.Add(this.hand_wmax_ctrl);
            this.RobotParamBox.Controls.Add(this.hand_wmin_ctrl);
            this.RobotParamBox.Controls.Add(this.hand_pmin_ctrl);
            this.RobotParamBox.Controls.Add(this.hand_pmax_ctrl);
            this.RobotParamBox.Controls.Add(this.label7);
            this.RobotParamBox.Controls.Add(this.wristRot_wmax_ctrl);
            this.RobotParamBox.Controls.Add(this.wristRot_wmin_ctrl);
            this.RobotParamBox.Controls.Add(this.Label18);
            this.RobotParamBox.Controls.Add(this.wristRot_pmin_ctrl);
            this.RobotParamBox.Controls.Add(this.wristRot_pmax_ctrl);
            this.RobotParamBox.Controls.Add(this.elbow_wmax_ctrl);
            this.RobotParamBox.Controls.Add(this.elbow_wmin_ctrl);
            this.RobotParamBox.Controls.Add(this.elbow_pmin_ctrl);
            this.RobotParamBox.Controls.Add(this.elbow_pmax_ctrl);
            this.RobotParamBox.Controls.Add(this.Label20);
            this.RobotParamBox.Controls.Add(this.shoulder_wmax_ctrl);
            this.RobotParamBox.Controls.Add(this.shoulder_wmin_ctrl);
            this.RobotParamBox.Controls.Add(this.shoulder_pmin_ctrl);
            this.RobotParamBox.Controls.Add(this.shoulder_pmax_ctrl);
            this.RobotParamBox.Controls.Add(this.Label21);
            this.RobotParamBox.Controls.Add(this.Label19);
            this.RobotParamBox.Controls.Add(this.wristFlex_wmax_ctrl);
            this.RobotParamBox.Controls.Add(this.wristFlex_wmin_ctrl);
            this.RobotParamBox.Controls.Add(this.wristFlex_pmin_ctrl);
            this.RobotParamBox.Controls.Add(this.wristFlex_pmax_ctrl);
            this.RobotParamBox.Controls.Add(this.label5);
            this.RobotParamBox.Controls.Add(this.label6);
            this.RobotParamBox.Controls.Add(this.label11);
            this.RobotParamBox.Controls.Add(this.label22);
            this.RobotParamBox.Enabled = false;
            this.RobotParamBox.Location = new System.Drawing.Point(4, 161);
            this.RobotParamBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotParamBox.Name = "RobotParamBox";
            this.RobotParamBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.RobotParamBox.Size = new System.Drawing.Size(492, 187);
            this.RobotParamBox.TabIndex = 138;
            this.RobotParamBox.TabStop = false;
            this.RobotParamBox.Text = "Joint Limits (Position, Velocity)";
            // 
            // hand_wmax_ctrl
            // 
            this.hand_wmax_ctrl.Location = new System.Drawing.Point(419, 153);
            this.hand_wmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_wmax_ctrl.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.hand_wmax_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.hand_wmax_ctrl.Name = "hand_wmax_ctrl";
            this.hand_wmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.hand_wmax_ctrl.TabIndex = 149;
            this.hand_wmax_ctrl.Value = new decimal(new int[] {
            90,
            0,
            0,
            0});
            this.hand_wmax_ctrl.ValueChanged += new System.EventHandler(this.hand_wmax_ctrl_ValueChanged);
            // 
            // hand_wmin_ctrl
            // 
            this.hand_wmin_ctrl.Location = new System.Drawing.Point(344, 153);
            this.hand_wmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_wmin_ctrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.hand_wmin_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.hand_wmin_ctrl.Name = "hand_wmin_ctrl";
            this.hand_wmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.hand_wmin_ctrl.TabIndex = 148;
            this.hand_wmin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.hand_wmin_ctrl.ValueChanged += new System.EventHandler(this.hand_wmin_ctrl_ValueChanged);
            // 
            // hand_pmin_ctrl
            // 
            this.hand_pmin_ctrl.Location = new System.Drawing.Point(197, 153);
            this.hand_pmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_pmin_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.hand_pmin_ctrl.Name = "hand_pmin_ctrl";
            this.hand_pmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.hand_pmin_ctrl.TabIndex = 146;
            this.hand_pmin_ctrl.Value = new decimal(new int[] {
            1928,
            0,
            0,
            0});
            this.hand_pmin_ctrl.ValueChanged += new System.EventHandler(this.hand_pmin_ctrl_ValueChanged);
            // 
            // hand_pmax_ctrl
            // 
            this.hand_pmax_ctrl.Location = new System.Drawing.Point(271, 153);
            this.hand_pmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_pmax_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.hand_pmax_ctrl.Name = "hand_pmax_ctrl";
            this.hand_pmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.hand_pmax_ctrl.TabIndex = 147;
            this.hand_pmax_ctrl.Value = new decimal(new int[] {
            2800,
            0,
            0,
            0});
            this.hand_pmax_ctrl.ValueChanged += new System.EventHandler(this.hand_pmax_ctrl_ValueChanged);
            // 
            // label7
            // 
            this.label7.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label7.Location = new System.Drawing.Point(5, 155);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(133, 18);
            this.label7.TabIndex = 145;
            this.label7.Text = "Hand Open/Close:";
            // 
            // wristRot_wmax_ctrl
            // 
            this.wristRot_wmax_ctrl.Location = new System.Drawing.Point(419, 95);
            this.wristRot_wmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_wmax_ctrl.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.wristRot_wmax_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristRot_wmax_ctrl.Name = "wristRot_wmax_ctrl";
            this.wristRot_wmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristRot_wmax_ctrl.TabIndex = 144;
            this.wristRot_wmax_ctrl.Value = new decimal(new int[] {
            90,
            0,
            0,
            0});
            this.wristRot_wmax_ctrl.ValueChanged += new System.EventHandler(this.wristRot_wmax_ctrl_ValueChanged);
            // 
            // wristRot_wmin_ctrl
            // 
            this.wristRot_wmin_ctrl.Location = new System.Drawing.Point(344, 95);
            this.wristRot_wmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_wmin_ctrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.wristRot_wmin_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristRot_wmin_ctrl.Name = "wristRot_wmin_ctrl";
            this.wristRot_wmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristRot_wmin_ctrl.TabIndex = 143;
            this.wristRot_wmin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristRot_wmin_ctrl.ValueChanged += new System.EventHandler(this.wristRot_wmin_ctrl_ValueChanged);
            // 
            // Label18
            // 
            this.Label18.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label18.Location = new System.Drawing.Point(5, 97);
            this.Label18.Name = "Label18";
            this.Label18.Size = new System.Drawing.Size(173, 18);
            this.Label18.TabIndex = 142;
            this.Label18.Text = "Wrist Rotation CCW/CW:";
            // 
            // wristRot_pmin_ctrl
            // 
            this.wristRot_pmin_ctrl.Location = new System.Drawing.Point(197, 95);
            this.wristRot_pmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_pmin_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristRot_pmin_ctrl.Name = "wristRot_pmin_ctrl";
            this.wristRot_pmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristRot_pmin_ctrl.TabIndex = 140;
            this.wristRot_pmin_ctrl.Value = new decimal(new int[] {
            1028,
            0,
            0,
            0});
            this.wristRot_pmin_ctrl.ValueChanged += new System.EventHandler(this.wristRot_pmin_ctrl_ValueChanged);
            // 
            // wristRot_pmax_ctrl
            // 
            this.wristRot_pmax_ctrl.Location = new System.Drawing.Point(271, 95);
            this.wristRot_pmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_pmax_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristRot_pmax_ctrl.Name = "wristRot_pmax_ctrl";
            this.wristRot_pmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristRot_pmax_ctrl.TabIndex = 141;
            this.wristRot_pmax_ctrl.Value = new decimal(new int[] {
            3073,
            0,
            0,
            0});
            this.wristRot_pmax_ctrl.ValueChanged += new System.EventHandler(this.wristRot_pmax_ctrl_ValueChanged);
            // 
            // elbow_wmax_ctrl
            // 
            this.elbow_wmax_ctrl.Location = new System.Drawing.Point(419, 64);
            this.elbow_wmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_wmax_ctrl.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.elbow_wmax_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.elbow_wmax_ctrl.Name = "elbow_wmax_ctrl";
            this.elbow_wmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.elbow_wmax_ctrl.TabIndex = 139;
            this.elbow_wmax_ctrl.Value = new decimal(new int[] {
            45,
            0,
            0,
            0});
            this.elbow_wmax_ctrl.ValueChanged += new System.EventHandler(this.elbow_wmax_ctrl_ValueChanged);
            // 
            // elbow_wmin_ctrl
            // 
            this.elbow_wmin_ctrl.Location = new System.Drawing.Point(344, 65);
            this.elbow_wmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_wmin_ctrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.elbow_wmin_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.elbow_wmin_ctrl.Name = "elbow_wmin_ctrl";
            this.elbow_wmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.elbow_wmin_ctrl.TabIndex = 138;
            this.elbow_wmin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.elbow_wmin_ctrl.ValueChanged += new System.EventHandler(this.elbow_wmin_ctrl_ValueChanged);
            // 
            // elbow_pmin_ctrl
            // 
            this.elbow_pmin_ctrl.Location = new System.Drawing.Point(197, 66);
            this.elbow_pmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_pmin_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.elbow_pmin_ctrl.Name = "elbow_pmin_ctrl";
            this.elbow_pmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.elbow_pmin_ctrl.TabIndex = 136;
            this.elbow_pmin_ctrl.Value = new decimal(new int[] {
            1784,
            0,
            0,
            0});
            this.elbow_pmin_ctrl.ValueChanged += new System.EventHandler(this.elbow_pmin_ctrl_ValueChanged);
            // 
            // elbow_pmax_ctrl
            // 
            this.elbow_pmax_ctrl.Location = new System.Drawing.Point(271, 66);
            this.elbow_pmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_pmax_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.elbow_pmax_ctrl.Name = "elbow_pmax_ctrl";
            this.elbow_pmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.elbow_pmax_ctrl.TabIndex = 137;
            this.elbow_pmax_ctrl.Value = new decimal(new int[] {
            2570,
            0,
            0,
            0});
            this.elbow_pmax_ctrl.ValueChanged += new System.EventHandler(this.elbow_pmax_ctrl_ValueChanged);
            // 
            // Label20
            // 
            this.Label20.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label20.Location = new System.Drawing.Point(5, 68);
            this.Label20.Name = "Label20";
            this.Label20.Size = new System.Drawing.Size(171, 18);
            this.Label20.TabIndex = 135;
            this.Label20.Text = "Elbow Extension/Flexion:";
            // 
            // shoulder_wmax_ctrl
            // 
            this.shoulder_wmax_ctrl.Location = new System.Drawing.Point(419, 37);
            this.shoulder_wmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_wmax_ctrl.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.shoulder_wmax_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.shoulder_wmax_ctrl.Name = "shoulder_wmax_ctrl";
            this.shoulder_wmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.shoulder_wmax_ctrl.TabIndex = 134;
            this.shoulder_wmax_ctrl.Value = new decimal(new int[] {
            67,
            0,
            0,
            0});
            this.shoulder_wmax_ctrl.ValueChanged += new System.EventHandler(this.shoulder_wmax_ctrl_ValueChanged);
            // 
            // shoulder_wmin_ctrl
            // 
            this.shoulder_wmin_ctrl.Location = new System.Drawing.Point(344, 37);
            this.shoulder_wmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_wmin_ctrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.shoulder_wmin_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.shoulder_wmin_ctrl.Name = "shoulder_wmin_ctrl";
            this.shoulder_wmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.shoulder_wmin_ctrl.TabIndex = 133;
            this.shoulder_wmin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.shoulder_wmin_ctrl.ValueChanged += new System.EventHandler(this.shoulder_wmin_ctrl_ValueChanged);
            // 
            // shoulder_pmin_ctrl
            // 
            this.shoulder_pmin_ctrl.Location = new System.Drawing.Point(197, 37);
            this.shoulder_pmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_pmin_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.shoulder_pmin_ctrl.Name = "shoulder_pmin_ctrl";
            this.shoulder_pmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.shoulder_pmin_ctrl.TabIndex = 131;
            this.shoulder_pmin_ctrl.Value = new decimal(new int[] {
            1028,
            0,
            0,
            0});
            this.shoulder_pmin_ctrl.ValueChanged += new System.EventHandler(this.shoulder_pmin_ctrl_ValueChanged);
            // 
            // shoulder_pmax_ctrl
            // 
            this.shoulder_pmax_ctrl.Location = new System.Drawing.Point(271, 37);
            this.shoulder_pmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_pmax_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.shoulder_pmax_ctrl.Name = "shoulder_pmax_ctrl";
            this.shoulder_pmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.shoulder_pmax_ctrl.TabIndex = 132;
            this.shoulder_pmax_ctrl.Value = new decimal(new int[] {
            3073,
            0,
            0,
            0});
            this.shoulder_pmax_ctrl.ValueChanged += new System.EventHandler(this.shoulder_pmax_ctrl_ValueChanged);
            // 
            // Label21
            // 
            this.Label21.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label21.Location = new System.Drawing.Point(5, 39);
            this.Label21.Name = "Label21";
            this.Label21.Size = new System.Drawing.Size(199, 18);
            this.Label21.TabIndex = 130;
            this.Label21.Text = "Shoulder Rotation CCW/CW:";
            // 
            // Label19
            // 
            this.Label19.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.Label19.Location = new System.Drawing.Point(5, 127);
            this.Label19.Name = "Label19";
            this.Label19.Size = new System.Drawing.Size(173, 18);
            this.Label19.TabIndex = 129;
            this.Label19.Text = "Wrist Extension/Flexion";
            // 
            // wristFlex_wmax_ctrl
            // 
            this.wristFlex_wmax_ctrl.Location = new System.Drawing.Point(419, 126);
            this.wristFlex_wmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_wmax_ctrl.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.wristFlex_wmax_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristFlex_wmax_ctrl.Name = "wristFlex_wmax_ctrl";
            this.wristFlex_wmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristFlex_wmax_ctrl.TabIndex = 128;
            this.wristFlex_wmax_ctrl.Value = new decimal(new int[] {
            67,
            0,
            0,
            0});
            this.wristFlex_wmax_ctrl.ValueChanged += new System.EventHandler(this.wristFlex_wmax_ctrl_ValueChanged);
            // 
            // wristFlex_wmin_ctrl
            // 
            this.wristFlex_wmin_ctrl.Location = new System.Drawing.Point(344, 126);
            this.wristFlex_wmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_wmin_ctrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.wristFlex_wmin_ctrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristFlex_wmin_ctrl.Name = "wristFlex_wmin_ctrl";
            this.wristFlex_wmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristFlex_wmin_ctrl.TabIndex = 127;
            this.wristFlex_wmin_ctrl.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristFlex_wmin_ctrl.ValueChanged += new System.EventHandler(this.wristFlex_wmin_ctrl_ValueChanged);
            // 
            // wristFlex_pmin_ctrl
            // 
            this.wristFlex_pmin_ctrl.Location = new System.Drawing.Point(197, 126);
            this.wristFlex_pmin_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_pmin_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristFlex_pmin_ctrl.Name = "wristFlex_pmin_ctrl";
            this.wristFlex_pmin_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristFlex_pmin_ctrl.TabIndex = 125;
            this.wristFlex_pmin_ctrl.Value = new decimal(new int[] {
            790,
            0,
            0,
            0});
            this.wristFlex_pmin_ctrl.ValueChanged += new System.EventHandler(this.wristFlex_pmin_ctrl_ValueChanged);
            // 
            // wristFlex_pmax_ctrl
            // 
            this.wristFlex_pmax_ctrl.Location = new System.Drawing.Point(271, 126);
            this.wristFlex_pmax_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_pmax_ctrl.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristFlex_pmax_ctrl.Name = "wristFlex_pmax_ctrl";
            this.wristFlex_pmax_ctrl.Size = new System.Drawing.Size(64, 22);
            this.wristFlex_pmax_ctrl.TabIndex = 126;
            this.wristFlex_pmax_ctrl.Value = new decimal(new int[] {
            3328,
            0,
            0,
            0});
            this.wristFlex_pmax_ctrl.ValueChanged += new System.EventHandler(this.wristFlex_pmax_ctrl_ValueChanged);
            // 
            // label5
            // 
            this.label5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label5.Location = new System.Drawing.Point(193, 15);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(47, 18);
            this.label5.TabIndex = 123;
            this.label5.Text = "Pmin:";
            // 
            // label6
            // 
            this.label6.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label6.Location = new System.Drawing.Point(267, 15);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(68, 18);
            this.label6.TabIndex = 124;
            this.label6.Text = "Pmax:";
            // 
            // label11
            // 
            this.label11.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label11.Location = new System.Drawing.Point(341, 15);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(67, 18);
            this.label11.TabIndex = 121;
            this.label11.Text = "Vmin:";
            // 
            // label22
            // 
            this.label22.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label22.Location = new System.Drawing.Point(415, 15);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(55, 18);
            this.label22.TabIndex = 122;
            this.label22.Text = "Vmax:";
            // 
            // SimBox
            // 
            this.SimBox.Controls.Add(this.SIMdcBTN);
            this.SimBox.Controls.Add(this.SIMconnectBTN);
            this.SimBox.Controls.Add(this.openSim);
            this.SimBox.Controls.Add(this.sim_flag);
            this.SimBox.Location = new System.Drawing.Point(501, 456);
            this.SimBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SimBox.Name = "SimBox";
            this.SimBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SimBox.Size = new System.Drawing.Size(296, 84);
            this.SimBox.TabIndex = 138;
            this.SimBox.TabStop = false;
            this.SimBox.Text = "Simulator";
            this.SimBox.Visible = false;
            // 
            // SIMdcBTN
            // 
            this.SIMdcBTN.Enabled = false;
            this.SIMdcBTN.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.SIMdcBTN.Location = new System.Drawing.Point(195, 30);
            this.SIMdcBTN.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SIMdcBTN.Name = "SIMdcBTN";
            this.SIMdcBTN.Size = new System.Drawing.Size(93, 28);
            this.SIMdcBTN.TabIndex = 11;
            this.SIMdcBTN.Text = "Disconnect";
            this.SIMdcBTN.Click += new System.EventHandler(this.SIMdcBTN_Click);
            // 
            // SIMconnectBTN
            // 
            this.SIMconnectBTN.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.SIMconnectBTN.Location = new System.Drawing.Point(104, 30);
            this.SIMconnectBTN.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SIMconnectBTN.Name = "SIMconnectBTN";
            this.SIMconnectBTN.Size = new System.Drawing.Size(83, 28);
            this.SIMconnectBTN.TabIndex = 10;
            this.SIMconnectBTN.Text = "Connect";
            this.SIMconnectBTN.Click += new System.EventHandler(this.SIMconnectBTN_Click);
            // 
            // openSim
            // 
            this.openSim.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.openSim.Location = new System.Drawing.Point(16, 30);
            this.openSim.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.openSim.Name = "openSim";
            this.openSim.Size = new System.Drawing.Size(83, 28);
            this.openSim.TabIndex = 10;
            this.openSim.Text = "Launch";
            this.openSim.Click += new System.EventHandler(this.openSim_Click);
            // 
            // sim_flag
            // 
            this.sim_flag.AutoSize = true;
            this.sim_flag.Location = new System.Drawing.Point(16, 63);
            this.sim_flag.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.sim_flag.Name = "sim_flag";
            this.sim_flag.Size = new System.Drawing.Size(18, 17);
            this.sim_flag.TabIndex = 12;
            this.sim_flag.UseVisualStyleBackColor = true;
            this.sim_flag.Visible = false;
            // 
            // LEDbox
            // 
            this.LEDbox.Controls.Add(this.label36);
            this.LEDbox.Controls.Add(this.LEDdisconnect);
            this.LEDbox.Controls.Add(this.comboBox1);
            this.LEDbox.Controls.Add(this.LEDconnect);
            this.LEDbox.Enabled = false;
            this.LEDbox.Location = new System.Drawing.Point(1145, 526);
            this.LEDbox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDbox.Name = "LEDbox";
            this.LEDbox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDbox.Size = new System.Drawing.Size(296, 84);
            this.LEDbox.TabIndex = 139;
            this.LEDbox.TabStop = false;
            this.LEDbox.Text = "LED Display";
            // 
            // label36
            // 
            this.label36.AutoSize = true;
            this.label36.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label36.Location = new System.Drawing.Point(19, 25);
            this.label36.Name = "label36";
            this.label36.Size = new System.Drawing.Size(73, 17);
            this.label36.TabIndex = 64;
            this.label36.Text = "COM Port:";
            // 
            // LEDdisconnect
            // 
            this.LEDdisconnect.Enabled = false;
            this.LEDdisconnect.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.LEDdisconnect.Location = new System.Drawing.Point(193, 41);
            this.LEDdisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDdisconnect.Name = "LEDdisconnect";
            this.LEDdisconnect.Size = new System.Drawing.Size(93, 28);
            this.LEDdisconnect.TabIndex = 11;
            this.LEDdisconnect.Text = "Disconnect";
            this.LEDdisconnect.Click += new System.EventHandler(this.LEDdisconnect_Click);
            // 
            // comboBox1
            // 
            this.comboBox1.FormattingEnabled = true;
            this.comboBox1.Location = new System.Drawing.Point(11, 43);
            this.comboBox1.Margin = new System.Windows.Forms.Padding(4);
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(87, 24);
            this.comboBox1.TabIndex = 65;
            // 
            // LEDconnect
            // 
            this.LEDconnect.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.LEDconnect.Location = new System.Drawing.Point(104, 41);
            this.LEDconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDconnect.Name = "LEDconnect";
            this.LEDconnect.Size = new System.Drawing.Size(83, 28);
            this.LEDconnect.TabIndex = 10;
            this.LEDconnect.Text = "Connect";
            this.LEDconnect.Click += new System.EventHandler(this.LEDconnect_Click);
            // 
            // cmbSerialPorts
            // 
            this.cmbSerialPorts.FormattingEnabled = true;
            this.cmbSerialPorts.Location = new System.Drawing.Point(272, 22);
            this.cmbSerialPorts.Margin = new System.Windows.Forms.Padding(4);
            this.cmbSerialPorts.Name = "cmbSerialPorts";
            this.cmbSerialPorts.Size = new System.Drawing.Size(87, 24);
            this.cmbSerialPorts.TabIndex = 15;
            // 
            // Timer1
            // 
            this.Timer1.Interval = 30;
            this.Timer1.Tick += new System.EventHandler(this.Timer1_Tick);
            // 
            // Timer3
            // 
            this.Timer3.Interval = 30;
            this.Timer3.Tick += new System.EventHandler(this.Timer3_Tick);
            // 
            // OpenFileDialog1
            // 
            this.OpenFileDialog1.FileName = "OpenFileDialog1";
            // 
            // Timer2
            // 
            this.Timer2.Interval = 30;
            this.Timer2.Tick += new System.EventHandler(this.Timer2_Tick);
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.button4);
            this.groupBox2.Controls.Add(this.button5);
            this.groupBox2.Controls.Add(this.button6);
            this.groupBox2.Enabled = false;
            this.groupBox2.Location = new System.Drawing.Point(1145, 614);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Size = new System.Drawing.Size(296, 84);
            this.groupBox2.TabIndex = 142;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Virtual Gamepad";
            // 
            // button4
            // 
            this.button4.Enabled = false;
            this.button4.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.button4.Location = new System.Drawing.Point(195, 30);
            this.button4.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(93, 28);
            this.button4.TabIndex = 11;
            this.button4.Text = "Disconnect";
            // 
            // button5
            // 
            this.button5.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.button5.Location = new System.Drawing.Point(104, 30);
            this.button5.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(83, 28);
            this.button5.TabIndex = 10;
            this.button5.Text = "Connect";
            // 
            // button6
            // 
            this.button6.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.button6.Location = new System.Drawing.Point(16, 30);
            this.button6.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(83, 28);
            this.button6.TabIndex = 10;
            this.button6.Text = "Launch";
            // 
            // MLBox
            // 
            this.MLBox.Controls.Add(this.home_BTN);
            this.MLBox.Controls.Add(this.torque_off);
            this.MLBox.Controls.Add(this.torque_on);
            this.MLBox.Controls.Add(this.MLdisable);
            this.MLBox.Controls.Add(this.MLenable);
            this.MLBox.Controls.Add(this.ML_stop);
            this.MLBox.Controls.Add(this.ML_start);
            this.MLBox.Controls.Add(this.hand_w);
            this.MLBox.Controls.Add(this.hand_p);
            this.MLBox.Controls.Add(this.label40);
            this.MLBox.Controls.Add(this.wristRot_w);
            this.MLBox.Controls.Add(this.label42);
            this.MLBox.Controls.Add(this.wristRot_p);
            this.MLBox.Controls.Add(this.elbow_w);
            this.MLBox.Controls.Add(this.elbow_p);
            this.MLBox.Controls.Add(this.label53);
            this.MLBox.Controls.Add(this.shoulder_w);
            this.MLBox.Controls.Add(this.shoulder_p);
            this.MLBox.Controls.Add(this.label54);
            this.MLBox.Controls.Add(this.label55);
            this.MLBox.Controls.Add(this.wristFlex_w);
            this.MLBox.Controls.Add(this.wristFlex_p);
            this.MLBox.Controls.Add(this.label58);
            this.MLBox.Controls.Add(this.label60);
            this.MLBox.Location = new System.Drawing.Point(855, 513);
            this.MLBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MLBox.Name = "MLBox";
            this.MLBox.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MLBox.Size = new System.Drawing.Size(285, 270);
            this.MLBox.TabIndex = 143;
            this.MLBox.TabStop = false;
            this.MLBox.Text = "Machine Learning Interface";
            // 
            // home_BTN
            // 
            this.home_BTN.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.home_BTN.Location = new System.Drawing.Point(215, 231);
            this.home_BTN.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.home_BTN.Name = "home_BTN";
            this.home_BTN.Size = new System.Drawing.Size(63, 28);
            this.home_BTN.TabIndex = 161;
            this.home_BTN.Text = "Home";
            this.home_BTN.Click += new System.EventHandler(this.home_BTN_Click);
            // 
            // torque_off
            // 
            this.torque_off.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.torque_off.Location = new System.Drawing.Point(108, 231);
            this.torque_off.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.torque_off.Name = "torque_off";
            this.torque_off.Size = new System.Drawing.Size(101, 28);
            this.torque_off.TabIndex = 160;
            this.torque_off.Text = "Torque Off";
            this.torque_off.Click += new System.EventHandler(this.torque_off_Click);
            // 
            // torque_on
            // 
            this.torque_on.Enabled = false;
            this.torque_on.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.torque_on.Location = new System.Drawing.Point(9, 231);
            this.torque_on.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.torque_on.Name = "torque_on";
            this.torque_on.Size = new System.Drawing.Size(93, 28);
            this.torque_on.TabIndex = 159;
            this.torque_on.Text = "Torque On";
            this.torque_on.Click += new System.EventHandler(this.torque_on_Click);
            // 
            // MLdisable
            // 
            this.MLdisable.Enabled = false;
            this.MLdisable.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MLdisable.Location = new System.Drawing.Point(99, 26);
            this.MLdisable.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MLdisable.Name = "MLdisable";
            this.MLdisable.Size = new System.Drawing.Size(83, 28);
            this.MLdisable.TabIndex = 145;
            this.MLdisable.Text = "Disable";
            this.MLdisable.Click += new System.EventHandler(this.MLdisable_Click);
            // 
            // MLenable
            // 
            this.MLenable.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.MLenable.Location = new System.Drawing.Point(9, 26);
            this.MLenable.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MLenable.Name = "MLenable";
            this.MLenable.Size = new System.Drawing.Size(83, 28);
            this.MLenable.TabIndex = 144;
            this.MLenable.Text = "Enable";
            this.MLenable.Click += new System.EventHandler(this.MLenable_Click);
            // 
            // ML_stop
            // 
            this.ML_stop.Location = new System.Drawing.Point(232, 80);
            this.ML_stop.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ML_stop.Name = "ML_stop";
            this.ML_stop.Size = new System.Drawing.Size(45, 23);
            this.ML_stop.TabIndex = 150;
            this.ML_stop.Text = "stop";
            this.ML_stop.UseVisualStyleBackColor = true;
            this.ML_stop.Click += new System.EventHandler(this.ML_stop_Click);
            // 
            // ML_start
            // 
            this.ML_start.Location = new System.Drawing.Point(187, 80);
            this.ML_start.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ML_start.Name = "ML_start";
            this.ML_start.Size = new System.Drawing.Size(45, 23);
            this.ML_start.TabIndex = 149;
            this.ML_start.Text = "start";
            this.ML_start.UseVisualStyleBackColor = true;
            this.ML_start.Click += new System.EventHandler(this.ML_start_Click);
            // 
            // hand_w
            // 
            this.hand_w.Location = new System.Drawing.Point(133, 198);
            this.hand_w.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_w.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.hand_w.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.hand_w.Name = "hand_w";
            this.hand_w.Size = new System.Drawing.Size(51, 22);
            this.hand_w.TabIndex = 148;
            this.hand_w.Value = new decimal(new int[] {
            90,
            0,
            0,
            0});
            // 
            // hand_p
            // 
            this.hand_p.Location = new System.Drawing.Point(80, 198);
            this.hand_p.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.hand_p.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.hand_p.Name = "hand_p";
            this.hand_p.Size = new System.Drawing.Size(51, 22);
            this.hand_p.TabIndex = 146;
            this.hand_p.Value = new decimal(new int[] {
            250,
            0,
            0,
            0});
            // 
            // label40
            // 
            this.label40.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label40.Location = new System.Drawing.Point(5, 199);
            this.label40.Name = "label40";
            this.label40.Size = new System.Drawing.Size(47, 18);
            this.label40.TabIndex = 145;
            this.label40.Text = "Hand:";
            // 
            // wristRot_w
            // 
            this.wristRot_w.Location = new System.Drawing.Point(133, 139);
            this.wristRot_w.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_w.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.wristRot_w.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristRot_w.Name = "wristRot_w";
            this.wristRot_w.Size = new System.Drawing.Size(51, 22);
            this.wristRot_w.TabIndex = 143;
            this.wristRot_w.Value = new decimal(new int[] {
            90,
            0,
            0,
            0});
            // 
            // label42
            // 
            this.label42.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label42.Location = new System.Drawing.Point(5, 142);
            this.label42.Name = "label42";
            this.label42.Size = new System.Drawing.Size(73, 18);
            this.label42.TabIndex = 142;
            this.label42.Text = "Wrist Rot:";
            // 
            // wristRot_p
            // 
            this.wristRot_p.Location = new System.Drawing.Point(80, 139);
            this.wristRot_p.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristRot_p.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristRot_p.Name = "wristRot_p";
            this.wristRot_p.Size = new System.Drawing.Size(51, 22);
            this.wristRot_p.TabIndex = 140;
            this.wristRot_p.Value = new decimal(new int[] {
            2048,
            0,
            0,
            0});
            // 
            // elbow_w
            // 
            this.elbow_w.Location = new System.Drawing.Point(133, 110);
            this.elbow_w.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_w.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.elbow_w.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.elbow_w.Name = "elbow_w";
            this.elbow_w.Size = new System.Drawing.Size(51, 22);
            this.elbow_w.TabIndex = 138;
            this.elbow_w.Value = new decimal(new int[] {
            45,
            0,
            0,
            0});
            // 
            // elbow_p
            // 
            this.elbow_p.Location = new System.Drawing.Point(80, 110);
            this.elbow_p.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.elbow_p.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.elbow_p.Name = "elbow_p";
            this.elbow_p.Size = new System.Drawing.Size(51, 22);
            this.elbow_p.TabIndex = 136;
            this.elbow_p.Value = new decimal(new int[] {
            2250,
            0,
            0,
            0});
            // 
            // label53
            // 
            this.label53.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label53.Location = new System.Drawing.Point(5, 111);
            this.label53.Name = "label53";
            this.label53.Size = new System.Drawing.Size(73, 18);
            this.label53.TabIndex = 135;
            this.label53.Text = "Elbow:";
            // 
            // shoulder_w
            // 
            this.shoulder_w.Location = new System.Drawing.Point(133, 81);
            this.shoulder_w.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_w.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.shoulder_w.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.shoulder_w.Name = "shoulder_w";
            this.shoulder_w.Size = new System.Drawing.Size(51, 22);
            this.shoulder_w.TabIndex = 133;
            this.shoulder_w.Value = new decimal(new int[] {
            67,
            0,
            0,
            0});
            // 
            // shoulder_p
            // 
            this.shoulder_p.Location = new System.Drawing.Point(80, 81);
            this.shoulder_p.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.shoulder_p.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.shoulder_p.Name = "shoulder_p";
            this.shoulder_p.Size = new System.Drawing.Size(51, 22);
            this.shoulder_p.TabIndex = 131;
            this.shoulder_p.Value = new decimal(new int[] {
            2048,
            0,
            0,
            0});
            // 
            // label54
            // 
            this.label54.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label54.Location = new System.Drawing.Point(5, 82);
            this.label54.Name = "label54";
            this.label54.Size = new System.Drawing.Size(73, 18);
            this.label54.TabIndex = 130;
            this.label54.Text = "Shoulder:";
            // 
            // label55
            // 
            this.label55.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label55.Location = new System.Drawing.Point(5, 171);
            this.label55.Name = "label55";
            this.label55.Size = new System.Drawing.Size(73, 18);
            this.label55.TabIndex = 129;
            this.label55.Text = "Wrist Flex:";
            // 
            // wristFlex_w
            // 
            this.wristFlex_w.Location = new System.Drawing.Point(133, 169);
            this.wristFlex_w.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_w.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.wristFlex_w.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.wristFlex_w.Name = "wristFlex_w";
            this.wristFlex_w.Size = new System.Drawing.Size(51, 22);
            this.wristFlex_w.TabIndex = 127;
            this.wristFlex_w.Value = new decimal(new int[] {
            67,
            0,
            0,
            0});
            // 
            // wristFlex_p
            // 
            this.wristFlex_p.Location = new System.Drawing.Point(80, 169);
            this.wristFlex_p.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.wristFlex_p.Maximum = new decimal(new int[] {
            4095,
            0,
            0,
            0});
            this.wristFlex_p.Name = "wristFlex_p";
            this.wristFlex_p.Size = new System.Drawing.Size(51, 22);
            this.wristFlex_p.TabIndex = 125;
            this.wristFlex_p.Value = new decimal(new int[] {
            2048,
            0,
            0,
            0});
            // 
            // label58
            // 
            this.label58.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label58.Location = new System.Drawing.Point(76, 59);
            this.label58.Name = "label58";
            this.label58.Size = new System.Drawing.Size(47, 18);
            this.label58.TabIndex = 123;
            this.label58.Text = "Pos:";
            // 
            // label60
            // 
            this.label60.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label60.Location = new System.Drawing.Point(129, 59);
            this.label60.Name = "label60";
            this.label60.Size = new System.Drawing.Size(47, 18);
            this.label60.TabIndex = 121;
            this.label60.Text = "Vel:";
            // 
            // checkGuide
            // 
            this.checkGuide.AutoSize = true;
            this.checkGuide.BackColor = System.Drawing.Color.Transparent;
            this.checkGuide.Enabled = false;
            this.checkGuide.Location = new System.Drawing.Point(120, 124);
            this.checkGuide.Margin = new System.Windows.Forms.Padding(4);
            this.checkGuide.Name = "checkGuide";
            this.checkGuide.Size = new System.Drawing.Size(18, 17);
            this.checkGuide.TabIndex = 163;
            this.checkGuide.UseVisualStyleBackColor = false;
            // 
            // labelStickRightY
            // 
            this.labelStickRightY.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelStickRightY.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelStickRightY.Location = new System.Drawing.Point(252, 198);
            this.labelStickRightY.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelStickRightY.Name = "labelStickRightY";
            this.labelStickRightY.Size = new System.Drawing.Size(79, 18);
            this.labelStickRightY.TabIndex = 162;
            this.labelStickRightY.Text = "1.0";
            this.labelStickRightY.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelStickRightX
            // 
            this.labelStickRightX.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelStickRightX.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelStickRightX.Location = new System.Drawing.Point(252, 181);
            this.labelStickRightX.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelStickRightX.Name = "labelStickRightX";
            this.labelStickRightX.Size = new System.Drawing.Size(79, 18);
            this.labelStickRightX.TabIndex = 161;
            this.labelStickRightX.Text = "1.0";
            this.labelStickRightX.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelStickLeftY
            // 
            this.labelStickLeftY.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelStickLeftY.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelStickLeftY.Location = new System.Drawing.Point(252, 156);
            this.labelStickLeftY.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelStickLeftY.Name = "labelStickLeftY";
            this.labelStickLeftY.Size = new System.Drawing.Size(79, 18);
            this.labelStickLeftY.TabIndex = 160;
            this.labelStickLeftY.Text = "1.0";
            this.labelStickLeftY.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelStickLeftX
            // 
            this.labelStickLeftX.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelStickLeftX.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelStickLeftX.Location = new System.Drawing.Point(252, 138);
            this.labelStickLeftX.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelStickLeftX.Name = "labelStickLeftX";
            this.labelStickLeftX.Size = new System.Drawing.Size(79, 19);
            this.labelStickLeftX.TabIndex = 159;
            this.labelStickLeftX.Text = "1.0";
            this.labelStickLeftX.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelTriggerRight
            // 
            this.labelTriggerRight.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelTriggerRight.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTriggerRight.Location = new System.Drawing.Point(252, 112);
            this.labelTriggerRight.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTriggerRight.Name = "labelTriggerRight";
            this.labelTriggerRight.Size = new System.Drawing.Size(79, 19);
            this.labelTriggerRight.TabIndex = 158;
            this.labelTriggerRight.Text = "1.0";
            this.labelTriggerRight.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // labelTriggerLeft
            // 
            this.labelTriggerLeft.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.labelTriggerLeft.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelTriggerLeft.Location = new System.Drawing.Point(252, 95);
            this.labelTriggerLeft.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelTriggerLeft.Name = "labelTriggerLeft";
            this.labelTriggerLeft.Size = new System.Drawing.Size(79, 19);
            this.labelTriggerLeft.TabIndex = 157;
            this.labelTriggerLeft.Text = "1.0";
            this.labelTriggerLeft.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // checkDPadLeft
            // 
            this.checkDPadLeft.AutoSize = true;
            this.checkDPadLeft.BackColor = System.Drawing.Color.Transparent;
            this.checkDPadLeft.Enabled = false;
            this.checkDPadLeft.Location = new System.Drawing.Point(252, 73);
            this.checkDPadLeft.Margin = new System.Windows.Forms.Padding(4);
            this.checkDPadLeft.Name = "checkDPadLeft";
            this.checkDPadLeft.Size = new System.Drawing.Size(18, 17);
            this.checkDPadLeft.TabIndex = 156;
            this.checkDPadLeft.UseVisualStyleBackColor = false;
            // 
            // checkDPadDown
            // 
            this.checkDPadDown.AutoSize = true;
            this.checkDPadDown.BackColor = System.Drawing.Color.Transparent;
            this.checkDPadDown.Enabled = false;
            this.checkDPadDown.Location = new System.Drawing.Point(252, 53);
            this.checkDPadDown.Margin = new System.Windows.Forms.Padding(4);
            this.checkDPadDown.Name = "checkDPadDown";
            this.checkDPadDown.Size = new System.Drawing.Size(18, 17);
            this.checkDPadDown.TabIndex = 155;
            this.checkDPadDown.UseVisualStyleBackColor = false;
            // 
            // checkDPadRight
            // 
            this.checkDPadRight.AutoSize = true;
            this.checkDPadRight.BackColor = System.Drawing.Color.Transparent;
            this.checkDPadRight.Enabled = false;
            this.checkDPadRight.Location = new System.Drawing.Point(252, 33);
            this.checkDPadRight.Margin = new System.Windows.Forms.Padding(4);
            this.checkDPadRight.Name = "checkDPadRight";
            this.checkDPadRight.Size = new System.Drawing.Size(18, 17);
            this.checkDPadRight.TabIndex = 154;
            this.checkDPadRight.UseVisualStyleBackColor = false;
            // 
            // checkDPadUp
            // 
            this.checkDPadUp.AutoSize = true;
            this.checkDPadUp.BackColor = System.Drawing.Color.Transparent;
            this.checkDPadUp.Enabled = false;
            this.checkDPadUp.Location = new System.Drawing.Point(252, 15);
            this.checkDPadUp.Margin = new System.Windows.Forms.Padding(4);
            this.checkDPadUp.Name = "checkDPadUp";
            this.checkDPadUp.Size = new System.Drawing.Size(18, 17);
            this.checkDPadUp.TabIndex = 153;
            this.checkDPadUp.UseVisualStyleBackColor = false;
            // 
            // checkStickLeft
            // 
            this.checkStickLeft.AutoSize = true;
            this.checkStickLeft.BackColor = System.Drawing.Color.Transparent;
            this.checkStickLeft.Enabled = false;
            this.checkStickLeft.Location = new System.Drawing.Point(120, 145);
            this.checkStickLeft.Margin = new System.Windows.Forms.Padding(4);
            this.checkStickLeft.Name = "checkStickLeft";
            this.checkStickLeft.Size = new System.Drawing.Size(18, 17);
            this.checkStickLeft.TabIndex = 152;
            this.checkStickLeft.UseVisualStyleBackColor = false;
            // 
            // checkStickRight
            // 
            this.checkStickRight.AutoSize = true;
            this.checkStickRight.BackColor = System.Drawing.Color.Transparent;
            this.checkStickRight.Enabled = false;
            this.checkStickRight.Location = new System.Drawing.Point(120, 164);
            this.checkStickRight.Margin = new System.Windows.Forms.Padding(4);
            this.checkStickRight.Name = "checkStickRight";
            this.checkStickRight.Size = new System.Drawing.Size(18, 17);
            this.checkStickRight.TabIndex = 151;
            this.checkStickRight.UseVisualStyleBackColor = false;
            // 
            // checkBack
            // 
            this.checkBack.AutoSize = true;
            this.checkBack.BackColor = System.Drawing.Color.Transparent;
            this.checkBack.Enabled = false;
            this.checkBack.Location = new System.Drawing.Point(120, 106);
            this.checkBack.Margin = new System.Windows.Forms.Padding(4);
            this.checkBack.Name = "checkBack";
            this.checkBack.Size = new System.Drawing.Size(18, 17);
            this.checkBack.TabIndex = 150;
            this.checkBack.UseVisualStyleBackColor = false;
            // 
            // checkStart
            // 
            this.checkStart.AutoSize = true;
            this.checkStart.BackColor = System.Drawing.Color.Transparent;
            this.checkStart.Enabled = false;
            this.checkStart.Location = new System.Drawing.Point(120, 86);
            this.checkStart.Margin = new System.Windows.Forms.Padding(4);
            this.checkStart.Name = "checkStart";
            this.checkStart.Size = new System.Drawing.Size(18, 17);
            this.checkStart.TabIndex = 149;
            this.checkStart.UseVisualStyleBackColor = false;
            // 
            // checkA
            // 
            this.checkA.AutoSize = true;
            this.checkA.BackColor = System.Drawing.Color.Transparent;
            this.checkA.Enabled = false;
            this.checkA.Location = new System.Drawing.Point(120, 15);
            this.checkA.Margin = new System.Windows.Forms.Padding(4);
            this.checkA.Name = "checkA";
            this.checkA.Size = new System.Drawing.Size(18, 17);
            this.checkA.TabIndex = 148;
            this.checkA.UseVisualStyleBackColor = false;
            // 
            // checkB
            // 
            this.checkB.AutoSize = true;
            this.checkB.BackColor = System.Drawing.Color.Transparent;
            this.checkB.Enabled = false;
            this.checkB.Location = new System.Drawing.Point(120, 32);
            this.checkB.Margin = new System.Windows.Forms.Padding(4);
            this.checkB.Name = "checkB";
            this.checkB.Size = new System.Drawing.Size(18, 17);
            this.checkB.TabIndex = 147;
            this.checkB.UseVisualStyleBackColor = false;
            // 
            // checkX
            // 
            this.checkX.AutoSize = true;
            this.checkX.BackColor = System.Drawing.Color.Transparent;
            this.checkX.Enabled = false;
            this.checkX.Location = new System.Drawing.Point(120, 49);
            this.checkX.Margin = new System.Windows.Forms.Padding(4);
            this.checkX.Name = "checkX";
            this.checkX.Size = new System.Drawing.Size(18, 17);
            this.checkX.TabIndex = 146;
            this.checkX.UseVisualStyleBackColor = false;
            // 
            // checkY
            // 
            this.checkY.AutoSize = true;
            this.checkY.BackColor = System.Drawing.Color.Transparent;
            this.checkY.Enabled = false;
            this.checkY.Location = new System.Drawing.Point(120, 68);
            this.checkY.Margin = new System.Windows.Forms.Padding(4);
            this.checkY.Name = "checkY";
            this.checkY.Size = new System.Drawing.Size(18, 17);
            this.checkY.TabIndex = 145;
            this.checkY.UseVisualStyleBackColor = false;
            // 
            // checkShoulderRight
            // 
            this.checkShoulderRight.AutoSize = true;
            this.checkShoulderRight.BackColor = System.Drawing.Color.Transparent;
            this.checkShoulderRight.Enabled = false;
            this.checkShoulderRight.Location = new System.Drawing.Point(120, 202);
            this.checkShoulderRight.Margin = new System.Windows.Forms.Padding(4);
            this.checkShoulderRight.Name = "checkShoulderRight";
            this.checkShoulderRight.Size = new System.Drawing.Size(18, 17);
            this.checkShoulderRight.TabIndex = 144;
            this.checkShoulderRight.UseVisualStyleBackColor = false;
            // 
            // label59
            // 
            this.label59.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label59.Location = new System.Drawing.Point(92, 12);
            this.label59.Name = "label59";
            this.label59.Size = new System.Drawing.Size(28, 18);
            this.label59.TabIndex = 150;
            this.label59.Text = "A:";
            this.label59.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label61
            // 
            this.label61.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label61.Location = new System.Drawing.Point(92, 31);
            this.label61.Name = "label61";
            this.label61.Size = new System.Drawing.Size(28, 18);
            this.label61.TabIndex = 164;
            this.label61.Text = "B:";
            this.label61.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label62
            // 
            this.label62.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label62.Location = new System.Drawing.Point(92, 47);
            this.label62.Name = "label62";
            this.label62.Size = new System.Drawing.Size(28, 18);
            this.label62.TabIndex = 165;
            this.label62.Text = "X:";
            this.label62.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label69
            // 
            this.label69.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label69.Location = new System.Drawing.Point(92, 66);
            this.label69.Name = "label69";
            this.label69.Size = new System.Drawing.Size(28, 18);
            this.label69.TabIndex = 166;
            this.label69.Text = "Y:";
            this.label69.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label71
            // 
            this.label71.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label71.Location = new System.Drawing.Point(45, 143);
            this.label71.Name = "label71";
            this.label71.Size = new System.Drawing.Size(75, 18);
            this.label71.TabIndex = 167;
            this.label71.Text = "StickLeft:";
            this.label71.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label73
            // 
            this.label73.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label73.Location = new System.Drawing.Point(32, 162);
            this.label73.Name = "label73";
            this.label73.Size = new System.Drawing.Size(88, 18);
            this.label73.TabIndex = 168;
            this.label73.Text = "StickRight:";
            this.label73.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label76
            // 
            this.label76.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label76.Location = new System.Drawing.Point(27, 181);
            this.label76.Name = "label76";
            this.label76.Size = new System.Drawing.Size(93, 18);
            this.label76.TabIndex = 169;
            this.label76.Text = "ShoulderLeft:";
            this.label76.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label80
            // 
            this.label80.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label80.Location = new System.Drawing.Point(13, 201);
            this.label80.Name = "label80";
            this.label80.Size = new System.Drawing.Size(107, 18);
            this.label80.TabIndex = 170;
            this.label80.Text = "ShoulderRight:";
            this.label80.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label81
            // 
            this.label81.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label81.Location = new System.Drawing.Point(145, 12);
            this.label81.Name = "label81";
            this.label81.Size = new System.Drawing.Size(107, 18);
            this.label81.TabIndex = 171;
            this.label81.Text = "DPadUp:";
            this.label81.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label82
            // 
            this.label82.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label82.Location = new System.Drawing.Point(145, 31);
            this.label82.Name = "label82";
            this.label82.Size = new System.Drawing.Size(107, 18);
            this.label82.TabIndex = 172;
            this.label82.Text = "DPadRight:";
            this.label82.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label84
            // 
            this.label84.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label84.Location = new System.Drawing.Point(145, 50);
            this.label84.Name = "label84";
            this.label84.Size = new System.Drawing.Size(107, 18);
            this.label84.TabIndex = 173;
            this.label84.Text = "DPadDown:";
            this.label84.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label85
            // 
            this.label85.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label85.Location = new System.Drawing.Point(145, 70);
            this.label85.Name = "label85";
            this.label85.Size = new System.Drawing.Size(107, 18);
            this.label85.TabIndex = 174;
            this.label85.Text = "DPadLeft:";
            this.label85.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label86
            // 
            this.label86.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label86.Location = new System.Drawing.Point(27, 124);
            this.label86.Name = "label86";
            this.label86.Size = new System.Drawing.Size(93, 18);
            this.label86.TabIndex = 177;
            this.label86.Text = "Guide:";
            this.label86.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label87
            // 
            this.label87.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label87.Location = new System.Drawing.Point(32, 106);
            this.label87.Name = "label87";
            this.label87.Size = new System.Drawing.Size(88, 18);
            this.label87.TabIndex = 176;
            this.label87.Text = "Back:";
            this.label87.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label88
            // 
            this.label88.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label88.Location = new System.Drawing.Point(45, 86);
            this.label88.Name = "label88";
            this.label88.Size = new System.Drawing.Size(75, 18);
            this.label88.TabIndex = 175;
            this.label88.Text = "Start:";
            this.label88.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // checkShoulderLeft
            // 
            this.checkShoulderLeft.AutoSize = true;
            this.checkShoulderLeft.BackColor = System.Drawing.Color.Transparent;
            this.checkShoulderLeft.CheckAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.checkShoulderLeft.Enabled = false;
            this.checkShoulderLeft.Location = new System.Drawing.Point(119, 183);
            this.checkShoulderLeft.Margin = new System.Windows.Forms.Padding(4);
            this.checkShoulderLeft.Name = "checkShoulderLeft";
            this.checkShoulderLeft.Size = new System.Drawing.Size(18, 17);
            this.checkShoulderLeft.TabIndex = 178;
            this.checkShoulderLeft.UseVisualStyleBackColor = false;
            // 
            // label105
            // 
            this.label105.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label105.Location = new System.Drawing.Point(144, 156);
            this.label105.Name = "label105";
            this.label105.Size = new System.Drawing.Size(107, 18);
            this.label105.TabIndex = 182;
            this.label105.Text = "StickLeftY:";
            this.label105.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label111
            // 
            this.label111.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label111.Location = new System.Drawing.Point(145, 138);
            this.label111.Name = "label111";
            this.label111.Size = new System.Drawing.Size(107, 18);
            this.label111.TabIndex = 181;
            this.label111.Text = "StickLeftX:";
            this.label111.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label112
            // 
            this.label112.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label112.Location = new System.Drawing.Point(145, 112);
            this.label112.Name = "label112";
            this.label112.Size = new System.Drawing.Size(107, 18);
            this.label112.TabIndex = 180;
            this.label112.Text = "TriggerRight:";
            this.label112.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label113
            // 
            this.label113.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label113.Location = new System.Drawing.Point(145, 94);
            this.label113.Name = "label113";
            this.label113.Size = new System.Drawing.Size(107, 18);
            this.label113.TabIndex = 179;
            this.label113.Text = "TriggerLeft:";
            this.label113.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label114
            // 
            this.label114.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label114.Location = new System.Drawing.Point(144, 197);
            this.label114.Name = "label114";
            this.label114.Size = new System.Drawing.Size(107, 18);
            this.label114.TabIndex = 184;
            this.label114.Text = "StickRightY:";
            this.label114.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label115
            // 
            this.label115.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label115.Location = new System.Drawing.Point(144, 178);
            this.label115.Name = "label115";
            this.label115.Size = new System.Drawing.Size(107, 18);
            this.label115.TabIndex = 183;
            this.label115.Text = "StickRightX:";
            this.label115.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // pollingWorker
            // 
            this.pollingWorker.WorkerSupportsCancellation = true;
            this.pollingWorker.DoWork += new System.ComponentModel.DoWorkEventHandler(this.pollingWorker_DoWork);
            // 
            // dynaConnect
            // 
            this.dynaConnect.Location = new System.Drawing.Point(8, 22);
            this.dynaConnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.dynaConnect.Name = "dynaConnect";
            this.dynaConnect.Size = new System.Drawing.Size(75, 23);
            this.dynaConnect.TabIndex = 185;
            this.dynaConnect.Text = "Connect";
            this.dynaConnect.UseVisualStyleBackColor = true;
            this.dynaConnect.Click += new System.EventHandler(this.dynaConnect_Click);
            // 
            // dynaDisconnect
            // 
            this.dynaDisconnect.Enabled = false;
            this.dynaDisconnect.Location = new System.Drawing.Point(91, 22);
            this.dynaDisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.dynaDisconnect.Name = "dynaDisconnect";
            this.dynaDisconnect.Size = new System.Drawing.Size(99, 23);
            this.dynaDisconnect.TabIndex = 186;
            this.dynaDisconnect.Text = "Disconnect";
            this.dynaDisconnect.UseVisualStyleBackColor = true;
            this.dynaDisconnect.Click += new System.EventHandler(this.dynaDisconnect_Click);
            // 
            // TorqueOn
            // 
            this.TorqueOn.Location = new System.Drawing.Point(11, 25);
            this.TorqueOn.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.TorqueOn.Name = "TorqueOn";
            this.TorqueOn.Size = new System.Drawing.Size(105, 26);
            this.TorqueOn.TabIndex = 187;
            this.TorqueOn.Text = "Torque On";
            this.TorqueOn.UseVisualStyleBackColor = true;
            this.TorqueOn.Click += new System.EventHandler(this.TorqueOn_Click);
            // 
            // TorqueOff
            // 
            this.TorqueOff.Enabled = false;
            this.TorqueOff.Location = new System.Drawing.Point(112, 25);
            this.TorqueOff.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.TorqueOff.Name = "TorqueOff";
            this.TorqueOff.Size = new System.Drawing.Size(115, 26);
            this.TorqueOff.TabIndex = 188;
            this.TorqueOff.Text = "Torque Off";
            this.TorqueOff.UseVisualStyleBackColor = true;
            this.TorqueOff.Click += new System.EventHandler(this.TorqueOff_Click);
            // 
            // LEDon
            // 
            this.LEDon.Location = new System.Drawing.Point(503, 554);
            this.LEDon.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDon.Name = "LEDon";
            this.LEDon.Size = new System.Drawing.Size(75, 23);
            this.LEDon.TabIndex = 189;
            this.LEDon.Text = "LED On";
            this.LEDon.UseVisualStyleBackColor = true;
            this.LEDon.Visible = false;
            this.LEDon.Click += new System.EventHandler(this.LEDon_Click);
            // 
            // LEDoff
            // 
            this.LEDoff.Location = new System.Drawing.Point(573, 554);
            this.LEDoff.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.LEDoff.Name = "LEDoff";
            this.LEDoff.Size = new System.Drawing.Size(75, 23);
            this.LEDoff.TabIndex = 190;
            this.LEDoff.Text = "LED Off";
            this.LEDoff.UseVisualStyleBackColor = true;
            this.LEDoff.Visible = false;
            this.LEDoff.Click += new System.EventHandler(this.LEDoff_Click);
            // 
            // moveCW
            // 
            this.moveCW.Enabled = false;
            this.moveCW.Location = new System.Drawing.Point(503, 584);
            this.moveCW.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.moveCW.Name = "moveCW";
            this.moveCW.Size = new System.Drawing.Size(107, 23);
            this.moveCW.TabIndex = 191;
            this.moveCW.Text = "Close Hand";
            this.moveCW.UseVisualStyleBackColor = true;
            this.moveCW.Visible = false;
            this.moveCW.Click += new System.EventHandler(this.moveCW_Click);
            // 
            // moveCCW
            // 
            this.moveCCW.Enabled = false;
            this.moveCCW.Location = new System.Drawing.Point(618, 584);
            this.moveCCW.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.moveCCW.Name = "moveCCW";
            this.moveCCW.Size = new System.Drawing.Size(107, 23);
            this.moveCCW.TabIndex = 192;
            this.moveCCW.Text = "Open Hand";
            this.moveCCW.UseVisualStyleBackColor = true;
            this.moveCCW.Visible = false;
            this.moveCCW.Click += new System.EventHandler(this.moveCCW_Click);
            // 
            // label116
            // 
            this.label116.AutoSize = true;
            this.label116.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label116.Location = new System.Drawing.Point(195, 26);
            this.label116.Name = "label116";
            this.label116.Size = new System.Drawing.Size(73, 17);
            this.label116.TabIndex = 65;
            this.label116.Text = "COM Port:";
            // 
            // label117
            // 
            this.label117.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label117.Location = new System.Drawing.Point(788, 5);
            this.label117.Name = "label117";
            this.label117.Size = new System.Drawing.Size(111, 18);
            this.label117.TabIndex = 193;
            this.label117.Text = "Delay (ms):";
            this.label117.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // delay
            // 
            this.delay.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.delay.Location = new System.Drawing.Point(897, 5);
            this.delay.Name = "delay";
            this.delay.Size = new System.Drawing.Size(56, 18);
            this.delay.TabIndex = 194;
            this.delay.Text = "--";
            // 
            // label118
            // 
            this.label118.AutoSize = true;
            this.label118.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label118.Location = new System.Drawing.Point(503, 649);
            this.label118.Name = "label118";
            this.label118.Size = new System.Drawing.Size(95, 17);
            this.label118.TabIndex = 195;
            this.label118.Text = "Comm Result:";
            this.label118.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label118.Visible = false;
            // 
            // dynaCommResult
            // 
            this.dynaCommResult.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.dynaCommResult.Location = new System.Drawing.Point(595, 649);
            this.dynaCommResult.Name = "dynaCommResult";
            this.dynaCommResult.Size = new System.Drawing.Size(52, 18);
            this.dynaCommResult.TabIndex = 196;
            this.dynaCommResult.Text = "--";
            this.dynaCommResult.Visible = false;
            // 
            // label120
            // 
            this.label120.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label120.Location = new System.Drawing.Point(547, 669);
            this.label120.Name = "label120";
            this.label120.Size = new System.Drawing.Size(52, 18);
            this.label120.TabIndex = 198;
            this.label120.Text = "Error:";
            this.label120.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label120.Visible = false;
            // 
            // dynaError
            // 
            this.dynaError.AutoSize = true;
            this.dynaError.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.dynaError.Location = new System.Drawing.Point(595, 670);
            this.dynaError.Name = "dynaError";
            this.dynaError.Size = new System.Drawing.Size(18, 17);
            this.dynaError.TabIndex = 197;
            this.dynaError.Text = "--";
            this.dynaError.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.dynaError.Visible = false;
            // 
            // readFeedback
            // 
            this.readFeedback.Location = new System.Drawing.Point(503, 612);
            this.readFeedback.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.readFeedback.Name = "readFeedback";
            this.readFeedback.Size = new System.Drawing.Size(136, 23);
            this.readFeedback.TabIndex = 199;
            this.readFeedback.Text = "Read Feedback";
            this.readFeedback.UseVisualStyleBackColor = true;
            this.readFeedback.Visible = false;
            this.readFeedback.Click += new System.EventHandler(this.readFeedback_Click);
            // 
            // delay_max
            // 
            this.delay_max.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.delay_max.Location = new System.Drawing.Point(1432, 693);
            this.delay_max.Name = "delay_max";
            this.delay_max.Size = new System.Drawing.Size(52, 18);
            this.delay_max.TabIndex = 203;
            this.delay_max.Text = "0";
            this.delay_max.Visible = false;
            // 
            // label121
            // 
            this.label121.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label121.Location = new System.Drawing.Point(1300, 693);
            this.label121.Name = "label121";
            this.label121.Size = new System.Drawing.Size(127, 18);
            this.label121.TabIndex = 202;
            this.label121.Text = "MAX DELAY (ms):";
            this.label121.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label121.Visible = false;
            // 
            // label119
            // 
            this.label119.AutoSize = true;
            this.label119.Location = new System.Drawing.Point(1301, 725);
            this.label119.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label119.Name = "label119";
            this.label119.Size = new System.Drawing.Size(52, 17);
            this.label119.TabIndex = 204;
            this.label119.Text = "Status:";
            this.label119.Visible = false;
            // 
            // dynaStatus
            // 
            this.dynaStatus.AutoSize = true;
            this.dynaStatus.Location = new System.Drawing.Point(1351, 725);
            this.dynaStatus.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.dynaStatus.Name = "dynaStatus";
            this.dynaStatus.Size = new System.Drawing.Size(94, 17);
            this.dynaStatus.TabIndex = 205;
            this.dynaStatus.Text = "Disconnected";
            this.dynaStatus.Visible = false;
            // 
            // cmbSerialRefresh
            // 
            this.cmbSerialRefresh.Location = new System.Drawing.Point(367, 22);
            this.cmbSerialRefresh.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cmbSerialRefresh.Name = "cmbSerialRefresh";
            this.cmbSerialRefresh.Size = new System.Drawing.Size(69, 23);
            this.cmbSerialRefresh.TabIndex = 206;
            this.cmbSerialRefresh.Text = "Refresh";
            this.cmbSerialRefresh.UseVisualStyleBackColor = true;
            this.cmbSerialRefresh.Click += new System.EventHandler(this.cmbSerialRefresh_Click);
            // 
            // BentoGroupBox
            // 
            this.BentoGroupBox.Controls.Add(this.label160);
            this.BentoGroupBox.Controls.Add(this.BentoRun);
            this.BentoGroupBox.Controls.Add(this.BentoSuspend);
            this.BentoGroupBox.Controls.Add(this.TorqueOn);
            this.BentoGroupBox.Controls.Add(this.TorqueOff);
            this.BentoGroupBox.Enabled = false;
            this.BentoGroupBox.Location = new System.Drawing.Point(4, 6);
            this.BentoGroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.BentoGroupBox.Name = "BentoGroupBox";
            this.BentoGroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.BentoGroupBox.Size = new System.Drawing.Size(492, 149);
            this.BentoGroupBox.TabIndex = 207;
            this.BentoGroupBox.TabStop = false;
            this.BentoGroupBox.Text = "Main Controls";
            // 
            // LogPID_Enabled
            // 
            this.LogPID_Enabled.AutoSize = true;
            this.LogPID_Enabled.Location = new System.Drawing.Point(11, 85);
            this.LogPID_Enabled.Margin = new System.Windows.Forms.Padding(4);
            this.LogPID_Enabled.Name = "LogPID_Enabled";
            this.LogPID_Enabled.Size = new System.Drawing.Size(80, 21);
            this.LogPID_Enabled.TabIndex = 217;
            this.LogPID_Enabled.Text = "Log PID";
            this.LogPID_Enabled.UseVisualStyleBackColor = true;
            this.LogPID_Enabled.CheckedChanged += new System.EventHandler(this.LogPID_Enabled_CheckedChanged);
            // 
            // label160
            // 
            this.label160.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label160.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label160.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label160.Location = new System.Drawing.Point(8, 62);
            this.label160.Name = "label160";
            this.label160.Size = new System.Drawing.Size(475, 38);
            this.label160.TabIndex = 205;
            this.label160.Text = "Click \'Torque On\' to allow the arm to hold its position and click \'Run\' to \r\nconn" +
    "ect the input devices to the arm, so that it can move.";
            // 
            // BentoRun
            // 
            this.BentoRun.Enabled = false;
            this.BentoRun.Location = new System.Drawing.Point(248, 25);
            this.BentoRun.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoRun.Name = "BentoRun";
            this.BentoRun.Size = new System.Drawing.Size(105, 26);
            this.BentoRun.TabIndex = 200;
            this.BentoRun.Text = "Run";
            this.BentoRun.UseVisualStyleBackColor = true;
            this.BentoRun.Click += new System.EventHandler(this.BentoRun_Click);
            // 
            // BentoSuspend
            // 
            this.BentoSuspend.Enabled = false;
            this.BentoSuspend.Location = new System.Drawing.Point(348, 25);
            this.BentoSuspend.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoSuspend.Name = "BentoSuspend";
            this.BentoSuspend.Size = new System.Drawing.Size(115, 26);
            this.BentoSuspend.TabIndex = 201;
            this.BentoSuspend.Text = "Suspend";
            this.BentoSuspend.UseVisualStyleBackColor = true;
            this.BentoSuspend.Click += new System.EventHandler(this.BentoSuspend_Click);
            // 
            // xBoxGroupBox
            // 
            this.xBoxGroupBox.Controls.Add(this.label80);
            this.xBoxGroupBox.Controls.Add(this.checkShoulderRight);
            this.xBoxGroupBox.Controls.Add(this.checkY);
            this.xBoxGroupBox.Controls.Add(this.checkX);
            this.xBoxGroupBox.Controls.Add(this.checkB);
            this.xBoxGroupBox.Controls.Add(this.checkA);
            this.xBoxGroupBox.Controls.Add(this.checkStart);
            this.xBoxGroupBox.Controls.Add(this.checkBack);
            this.xBoxGroupBox.Controls.Add(this.checkStickRight);
            this.xBoxGroupBox.Controls.Add(this.checkStickLeft);
            this.xBoxGroupBox.Controls.Add(this.checkDPadUp);
            this.xBoxGroupBox.Controls.Add(this.checkDPadRight);
            this.xBoxGroupBox.Controls.Add(this.checkDPadDown);
            this.xBoxGroupBox.Controls.Add(this.checkDPadLeft);
            this.xBoxGroupBox.Controls.Add(this.labelTriggerLeft);
            this.xBoxGroupBox.Controls.Add(this.labelTriggerRight);
            this.xBoxGroupBox.Controls.Add(this.labelStickLeftX);
            this.xBoxGroupBox.Controls.Add(this.labelStickLeftY);
            this.xBoxGroupBox.Controls.Add(this.labelStickRightX);
            this.xBoxGroupBox.Controls.Add(this.label114);
            this.xBoxGroupBox.Controls.Add(this.labelStickRightY);
            this.xBoxGroupBox.Controls.Add(this.label115);
            this.xBoxGroupBox.Controls.Add(this.checkGuide);
            this.xBoxGroupBox.Controls.Add(this.label105);
            this.xBoxGroupBox.Controls.Add(this.label59);
            this.xBoxGroupBox.Controls.Add(this.label111);
            this.xBoxGroupBox.Controls.Add(this.label61);
            this.xBoxGroupBox.Controls.Add(this.label112);
            this.xBoxGroupBox.Controls.Add(this.label62);
            this.xBoxGroupBox.Controls.Add(this.label113);
            this.xBoxGroupBox.Controls.Add(this.label69);
            this.xBoxGroupBox.Controls.Add(this.checkShoulderLeft);
            this.xBoxGroupBox.Controls.Add(this.label71);
            this.xBoxGroupBox.Controls.Add(this.label86);
            this.xBoxGroupBox.Controls.Add(this.label73);
            this.xBoxGroupBox.Controls.Add(this.label87);
            this.xBoxGroupBox.Controls.Add(this.label76);
            this.xBoxGroupBox.Controls.Add(this.label88);
            this.xBoxGroupBox.Controls.Add(this.label81);
            this.xBoxGroupBox.Controls.Add(this.label85);
            this.xBoxGroupBox.Controls.Add(this.label82);
            this.xBoxGroupBox.Controls.Add(this.label84);
            this.xBoxGroupBox.Enabled = false;
            this.xBoxGroupBox.Location = new System.Drawing.Point(4, 4);
            this.xBoxGroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.xBoxGroupBox.Name = "xBoxGroupBox";
            this.xBoxGroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.xBoxGroupBox.Size = new System.Drawing.Size(349, 223);
            this.xBoxGroupBox.TabIndex = 208;
            this.xBoxGroupBox.TabStop = false;
            this.xBoxGroupBox.Text = "Xbox";
            // 
            // XboxDisconnect
            // 
            this.XboxDisconnect.Enabled = false;
            this.XboxDisconnect.Location = new System.Drawing.Point(87, 22);
            this.XboxDisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.XboxDisconnect.Name = "XboxDisconnect";
            this.XboxDisconnect.Size = new System.Drawing.Size(99, 23);
            this.XboxDisconnect.TabIndex = 210;
            this.XboxDisconnect.Text = "Disconnect";
            this.XboxDisconnect.UseVisualStyleBackColor = true;
            this.XboxDisconnect.Click += new System.EventHandler(this.XboxDisconnect_Click);
            // 
            // XboxConnect
            // 
            this.XboxConnect.Location = new System.Drawing.Point(7, 22);
            this.XboxConnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.XboxConnect.Name = "XboxConnect";
            this.XboxConnect.Size = new System.Drawing.Size(75, 23);
            this.XboxConnect.TabIndex = 209;
            this.XboxConnect.Text = "Connect";
            this.XboxConnect.UseVisualStyleBackColor = true;
            this.XboxConnect.Click += new System.EventHandler(this.XboxConnect_Click);
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.groupBox18);
            this.groupBox4.Controls.Add(this.groupBox17);
            this.groupBox4.Controls.Add(this.groupBox15);
            this.groupBox4.Controls.Add(this.groupBox7);
            this.groupBox4.Controls.Add(this.groupBox8);
            this.groupBox4.Controls.Add(this.groupBox5);
            this.groupBox4.Location = new System.Drawing.Point(4, 7);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox4.Size = new System.Drawing.Size(791, 743);
            this.groupBox4.TabIndex = 211;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Input Devices";
            // 
            // groupBox18
            // 
            this.groupBox18.Controls.Add(this.ArduinoInputCOM);
            this.groupBox18.Controls.Add(this.label204);
            this.groupBox18.Controls.Add(this.ArduinoInputClearAll);
            this.groupBox18.Controls.Add(this.ArduinoInputConnect);
            this.groupBox18.Controls.Add(this.ArduinoInputSelectAll);
            this.groupBox18.Controls.Add(this.ArduinoInputDisconnect);
            this.groupBox18.Controls.Add(this.ArduinoInputList);
            this.groupBox18.Location = new System.Drawing.Point(399, 18);
            this.groupBox18.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox18.Name = "groupBox18";
            this.groupBox18.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox18.Size = new System.Drawing.Size(383, 180);
            this.groupBox18.TabIndex = 227;
            this.groupBox18.TabStop = false;
            this.groupBox18.Text = "Arduino - Setup";
            // 
            // ArduinoInputCOM
            // 
            this.ArduinoInputCOM.FormattingEnabled = true;
            this.ArduinoInputCOM.Location = new System.Drawing.Point(85, 48);
            this.ArduinoInputCOM.Margin = new System.Windows.Forms.Padding(4);
            this.ArduinoInputCOM.Name = "ArduinoInputCOM";
            this.ArduinoInputCOM.Size = new System.Drawing.Size(97, 24);
            this.ArduinoInputCOM.TabIndex = 226;
            // 
            // label204
            // 
            this.label204.AutoSize = true;
            this.label204.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label204.Location = new System.Drawing.Point(8, 52);
            this.label204.Name = "label204";
            this.label204.Size = new System.Drawing.Size(73, 17);
            this.label204.TabIndex = 227;
            this.label204.Text = "COM Port:";
            // 
            // ArduinoInputClearAll
            // 
            this.ArduinoInputClearAll.Enabled = false;
            this.ArduinoInputClearAll.Location = new System.Drawing.Point(283, 145);
            this.ArduinoInputClearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ArduinoInputClearAll.Name = "ArduinoInputClearAll";
            this.ArduinoInputClearAll.Size = new System.Drawing.Size(88, 23);
            this.ArduinoInputClearAll.TabIndex = 225;
            this.ArduinoInputClearAll.Text = "Clear All";
            this.ArduinoInputClearAll.UseVisualStyleBackColor = true;
            this.ArduinoInputClearAll.Click += new System.EventHandler(this.ArduinoInputClearAll_Click);
            // 
            // ArduinoInputConnect
            // 
            this.ArduinoInputConnect.Location = new System.Drawing.Point(7, 22);
            this.ArduinoInputConnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ArduinoInputConnect.Name = "ArduinoInputConnect";
            this.ArduinoInputConnect.Size = new System.Drawing.Size(75, 23);
            this.ArduinoInputConnect.TabIndex = 209;
            this.ArduinoInputConnect.Text = "Connect";
            this.ArduinoInputConnect.UseVisualStyleBackColor = true;
            this.ArduinoInputConnect.Click += new System.EventHandler(this.ArduinoInputConnect_Click);
            // 
            // ArduinoInputSelectAll
            // 
            this.ArduinoInputSelectAll.Enabled = false;
            this.ArduinoInputSelectAll.Location = new System.Drawing.Point(189, 145);
            this.ArduinoInputSelectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ArduinoInputSelectAll.Name = "ArduinoInputSelectAll";
            this.ArduinoInputSelectAll.Size = new System.Drawing.Size(88, 23);
            this.ArduinoInputSelectAll.TabIndex = 224;
            this.ArduinoInputSelectAll.Text = "Select All";
            this.ArduinoInputSelectAll.UseVisualStyleBackColor = true;
            this.ArduinoInputSelectAll.Click += new System.EventHandler(this.ArduinoInputSelectAll_Click);
            // 
            // ArduinoInputDisconnect
            // 
            this.ArduinoInputDisconnect.Enabled = false;
            this.ArduinoInputDisconnect.Location = new System.Drawing.Point(87, 22);
            this.ArduinoInputDisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.ArduinoInputDisconnect.Name = "ArduinoInputDisconnect";
            this.ArduinoInputDisconnect.Size = new System.Drawing.Size(99, 23);
            this.ArduinoInputDisconnect.TabIndex = 210;
            this.ArduinoInputDisconnect.Text = "Disconnect";
            this.ArduinoInputDisconnect.UseVisualStyleBackColor = true;
            this.ArduinoInputDisconnect.Click += new System.EventHandler(this.ArduinoInputDisconnect_Click);
            // 
            // ArduinoInputList
            // 
            this.ArduinoInputList.CheckOnClick = true;
            this.ArduinoInputList.Enabled = false;
            this.ArduinoInputList.FormattingEnabled = true;
            this.ArduinoInputList.Items.AddRange(new object[] {
            "A0 - Arduino",
            "A1 - Arduino",
            "A2 - Arduino",
            "A3 - Arduino",
            "A4 - Arduino",
            "A5 - Arduino",
            "A6 - Arduino",
            "A7 - Arduino"});
            this.ArduinoInputList.Location = new System.Drawing.Point(192, 23);
            this.ArduinoInputList.Margin = new System.Windows.Forms.Padding(4);
            this.ArduinoInputList.Name = "ArduinoInputList";
            this.ArduinoInputList.Size = new System.Drawing.Size(177, 89);
            this.ArduinoInputList.TabIndex = 223;
            // 
            // groupBox17
            // 
            this.groupBox17.Controls.Add(this.SLRTclearAll);
            this.groupBox17.Controls.Add(this.SLRTconnect);
            this.groupBox17.Controls.Add(this.SLRTselectAll);
            this.groupBox17.Controls.Add(this.SLRTdisconnect);
            this.groupBox17.Controls.Add(this.SLRTlist);
            this.groupBox17.Location = new System.Drawing.Point(400, 206);
            this.groupBox17.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox17.Name = "groupBox17";
            this.groupBox17.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox17.Size = new System.Drawing.Size(383, 180);
            this.groupBox17.TabIndex = 226;
            this.groupBox17.TabStop = false;
            this.groupBox17.Text = "Simulink Realtime (xPC Target) - Setup";
            // 
            // SLRTclearAll
            // 
            this.SLRTclearAll.Enabled = false;
            this.SLRTclearAll.Location = new System.Drawing.Point(283, 145);
            this.SLRTclearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SLRTclearAll.Name = "SLRTclearAll";
            this.SLRTclearAll.Size = new System.Drawing.Size(88, 23);
            this.SLRTclearAll.TabIndex = 225;
            this.SLRTclearAll.Text = "Clear All";
            this.SLRTclearAll.UseVisualStyleBackColor = true;
            this.SLRTclearAll.Click += new System.EventHandler(this.SLRTclearAll_Click);
            // 
            // SLRTconnect
            // 
            this.SLRTconnect.Location = new System.Drawing.Point(7, 22);
            this.SLRTconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SLRTconnect.Name = "SLRTconnect";
            this.SLRTconnect.Size = new System.Drawing.Size(75, 23);
            this.SLRTconnect.TabIndex = 209;
            this.SLRTconnect.Text = "Connect";
            this.SLRTconnect.UseVisualStyleBackColor = true;
            this.SLRTconnect.Click += new System.EventHandler(this.SLRTconnect_Click);
            // 
            // SLRTselectAll
            // 
            this.SLRTselectAll.Enabled = false;
            this.SLRTselectAll.Location = new System.Drawing.Point(189, 145);
            this.SLRTselectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SLRTselectAll.Name = "SLRTselectAll";
            this.SLRTselectAll.Size = new System.Drawing.Size(88, 23);
            this.SLRTselectAll.TabIndex = 224;
            this.SLRTselectAll.Text = "Select All";
            this.SLRTselectAll.UseVisualStyleBackColor = true;
            this.SLRTselectAll.Click += new System.EventHandler(this.SLRTselectAll_Click);
            // 
            // SLRTdisconnect
            // 
            this.SLRTdisconnect.Enabled = false;
            this.SLRTdisconnect.Location = new System.Drawing.Point(87, 22);
            this.SLRTdisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.SLRTdisconnect.Name = "SLRTdisconnect";
            this.SLRTdisconnect.Size = new System.Drawing.Size(99, 23);
            this.SLRTdisconnect.TabIndex = 210;
            this.SLRTdisconnect.Text = "Disconnect";
            this.SLRTdisconnect.UseVisualStyleBackColor = true;
            this.SLRTdisconnect.Click += new System.EventHandler(this.SLRTdisconnect_Click);
            // 
            // SLRTlist
            // 
            this.SLRTlist.CheckOnClick = true;
            this.SLRTlist.Enabled = false;
            this.SLRTlist.FormattingEnabled = true;
            this.SLRTlist.Items.AddRange(new object[] {
            "Ch1 - SLRT",
            "Ch2 - SLRT",
            "Ch3 - SLRT",
            "Ch4 - SLRT",
            "Ch5 - SLRT",
            "Ch6 - SLRT",
            "Ch7 - SLRT",
            "Ch8 - SLRT"});
            this.SLRTlist.Location = new System.Drawing.Point(192, 23);
            this.SLRTlist.Margin = new System.Windows.Forms.Padding(4);
            this.SLRTlist.Name = "SLRTlist";
            this.SLRTlist.Size = new System.Drawing.Size(177, 89);
            this.SLRTlist.TabIndex = 223;
            // 
            // groupBox15
            // 
            this.groupBox15.Controls.Add(this.biopatrecMode);
            this.groupBox15.Controls.Add(this.label202);
            this.groupBox15.Controls.Add(this.biopatrecIPport);
            this.groupBox15.Controls.Add(this.label186);
            this.groupBox15.Controls.Add(this.label188);
            this.groupBox15.Controls.Add(this.biopatrecClearAll);
            this.groupBox15.Controls.Add(this.biopatrecConnect);
            this.groupBox15.Controls.Add(this.biopatrecIPaddr);
            this.groupBox15.Controls.Add(this.biopatrecSelectAll);
            this.groupBox15.Controls.Add(this.biopatrecDisconnect);
            this.groupBox15.Controls.Add(this.biopatrecList);
            this.groupBox15.Location = new System.Drawing.Point(8, 578);
            this.groupBox15.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox15.Name = "groupBox15";
            this.groupBox15.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox15.Size = new System.Drawing.Size(383, 160);
            this.groupBox15.TabIndex = 226;
            this.groupBox15.TabStop = false;
            this.groupBox15.Text = "BioPatRec - Setup";
            // 
            // biopatrecMode
            // 
            this.biopatrecMode.FormattingEnabled = true;
            this.biopatrecMode.Items.AddRange(new object[] {
            "Input",
            "Output - TAC"});
            this.biopatrecMode.Location = new System.Drawing.Point(65, 111);
            this.biopatrecMode.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecMode.Name = "biopatrecMode";
            this.biopatrecMode.Size = new System.Drawing.Size(116, 24);
            this.biopatrecMode.TabIndex = 232;
            // 
            // label202
            // 
            this.label202.AutoSize = true;
            this.label202.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label202.Location = new System.Drawing.Point(13, 114);
            this.label202.Name = "label202";
            this.label202.Size = new System.Drawing.Size(47, 17);
            this.label202.TabIndex = 231;
            this.label202.Text = "Mode:";
            // 
            // biopatrecIPport
            // 
            this.biopatrecIPport.Location = new System.Drawing.Point(65, 81);
            this.biopatrecIPport.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecIPport.Name = "biopatrecIPport";
            this.biopatrecIPport.Size = new System.Drawing.Size(51, 22);
            this.biopatrecIPport.TabIndex = 230;
            this.biopatrecIPport.Text = "30000";
            // 
            // label186
            // 
            this.label186.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label186.Location = new System.Drawing.Point(4, 55);
            this.label186.Name = "label186";
            this.label186.Size = new System.Drawing.Size(60, 15);
            this.label186.TabIndex = 228;
            this.label186.Text = "IP addr:";
            // 
            // label188
            // 
            this.label188.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label188.Location = new System.Drawing.Point(8, 85);
            this.label188.Name = "label188";
            this.label188.Size = new System.Drawing.Size(55, 21);
            this.label188.TabIndex = 229;
            this.label188.Text = "IP port:";
            // 
            // biopatrecClearAll
            // 
            this.biopatrecClearAll.Enabled = false;
            this.biopatrecClearAll.Location = new System.Drawing.Point(283, 133);
            this.biopatrecClearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecClearAll.Name = "biopatrecClearAll";
            this.biopatrecClearAll.Size = new System.Drawing.Size(88, 23);
            this.biopatrecClearAll.TabIndex = 225;
            this.biopatrecClearAll.Text = "Clear All";
            this.biopatrecClearAll.UseVisualStyleBackColor = true;
            this.biopatrecClearAll.Click += new System.EventHandler(this.biopatrecClearall_Click);
            // 
            // biopatrecConnect
            // 
            this.biopatrecConnect.Location = new System.Drawing.Point(7, 22);
            this.biopatrecConnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecConnect.Name = "biopatrecConnect";
            this.biopatrecConnect.Size = new System.Drawing.Size(75, 23);
            this.biopatrecConnect.TabIndex = 209;
            this.biopatrecConnect.Text = "Connect";
            this.biopatrecConnect.UseVisualStyleBackColor = true;
            this.biopatrecConnect.Click += new System.EventHandler(this.biopatrecConnect_Click);
            // 
            // biopatrecIPaddr
            // 
            this.biopatrecIPaddr.Location = new System.Drawing.Point(65, 52);
            this.biopatrecIPaddr.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecIPaddr.Name = "biopatrecIPaddr";
            this.biopatrecIPaddr.Size = new System.Drawing.Size(116, 22);
            this.biopatrecIPaddr.TabIndex = 227;
            this.biopatrecIPaddr.Text = "0.0.0.0";
            // 
            // biopatrecSelectAll
            // 
            this.biopatrecSelectAll.Enabled = false;
            this.biopatrecSelectAll.Location = new System.Drawing.Point(189, 133);
            this.biopatrecSelectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecSelectAll.Name = "biopatrecSelectAll";
            this.biopatrecSelectAll.Size = new System.Drawing.Size(88, 23);
            this.biopatrecSelectAll.TabIndex = 224;
            this.biopatrecSelectAll.Text = "Select All";
            this.biopatrecSelectAll.UseVisualStyleBackColor = true;
            this.biopatrecSelectAll.Click += new System.EventHandler(this.biopatrecSelectAll_Click);
            // 
            // biopatrecDisconnect
            // 
            this.biopatrecDisconnect.Enabled = false;
            this.biopatrecDisconnect.Location = new System.Drawing.Point(87, 22);
            this.biopatrecDisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.biopatrecDisconnect.Name = "biopatrecDisconnect";
            this.biopatrecDisconnect.Size = new System.Drawing.Size(99, 23);
            this.biopatrecDisconnect.TabIndex = 210;
            this.biopatrecDisconnect.Text = "Disconnect";
            this.biopatrecDisconnect.UseVisualStyleBackColor = true;
            this.biopatrecDisconnect.Click += new System.EventHandler(this.biopatrecDisconnect_Click);
            // 
            // biopatrecList
            // 
            this.biopatrecList.CheckOnClick = true;
            this.biopatrecList.Enabled = false;
            this.biopatrecList.FormattingEnabled = true;
            this.biopatrecList.Items.AddRange(new object[] {
            "Rest",
            "Open Hand",
            "Close Hand",
            "Flex Hand",
            "Extend Hand",
            "Pronation",
            "Supination",
            "Thumb Extend",
            "Thumb Flex",
            "Index Extend",
            "Index Flex",
            "Middle Extend",
            "Middle Flex",
            "Ring Extend",
            "Ring Flex",
            "Little Extend",
            "Little Flex",
            "Point",
            "Thumb Yaw Flex",
            "Thumb Yaw Extend",
            "Flex Elbow",
            "Extend Elbow",
            "Rotate Elbow Int",
            "Rotate Elbow Ext",
            "Side Grip"});
            this.biopatrecList.Location = new System.Drawing.Point(192, 23);
            this.biopatrecList.Margin = new System.Windows.Forms.Padding(4);
            this.biopatrecList.Name = "biopatrecList";
            this.biopatrecList.Size = new System.Drawing.Size(177, 89);
            this.biopatrecList.TabIndex = 223;
            // 
            // groupBox7
            // 
            this.groupBox7.Controls.Add(this.pictureBox10);
            this.groupBox7.Controls.Add(this.MYOclearAll);
            this.groupBox7.Controls.Add(this.MYOconnect);
            this.groupBox7.Controls.Add(this.MYOselectAll);
            this.groupBox7.Controls.Add(this.MYOdisconnect);
            this.groupBox7.Controls.Add(this.MYOlist);
            this.groupBox7.Location = new System.Drawing.Point(8, 206);
            this.groupBox7.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox7.Size = new System.Drawing.Size(383, 180);
            this.groupBox7.TabIndex = 211;
            this.groupBox7.TabStop = false;
            this.groupBox7.Text = "MYO Armband - Setup";
            // 
            // pictureBox10
            // 
            this.pictureBox10.Image = global::brachIOplexus.Properties.Resources.myo_edit;
            this.pictureBox10.Location = new System.Drawing.Point(5, 52);
            this.pictureBox10.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox10.Name = "pictureBox10";
            this.pictureBox10.Size = new System.Drawing.Size(177, 116);
            this.pictureBox10.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox10.TabIndex = 223;
            this.pictureBox10.TabStop = false;
            // 
            // MYOclearAll
            // 
            this.MYOclearAll.Enabled = false;
            this.MYOclearAll.Location = new System.Drawing.Point(283, 145);
            this.MYOclearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MYOclearAll.Name = "MYOclearAll";
            this.MYOclearAll.Size = new System.Drawing.Size(88, 23);
            this.MYOclearAll.TabIndex = 225;
            this.MYOclearAll.Text = "Clear All";
            this.MYOclearAll.UseVisualStyleBackColor = true;
            this.MYOclearAll.Click += new System.EventHandler(this.MYOclearAll_Click);
            // 
            // MYOconnect
            // 
            this.MYOconnect.Location = new System.Drawing.Point(7, 22);
            this.MYOconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MYOconnect.Name = "MYOconnect";
            this.MYOconnect.Size = new System.Drawing.Size(75, 23);
            this.MYOconnect.TabIndex = 209;
            this.MYOconnect.Text = "Connect";
            this.MYOconnect.UseVisualStyleBackColor = true;
            this.MYOconnect.Click += new System.EventHandler(this.MYOconnect_Click);
            // 
            // MYOselectAll
            // 
            this.MYOselectAll.Enabled = false;
            this.MYOselectAll.Location = new System.Drawing.Point(189, 145);
            this.MYOselectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MYOselectAll.Name = "MYOselectAll";
            this.MYOselectAll.Size = new System.Drawing.Size(88, 23);
            this.MYOselectAll.TabIndex = 224;
            this.MYOselectAll.Text = "Select All";
            this.MYOselectAll.UseVisualStyleBackColor = true;
            this.MYOselectAll.Click += new System.EventHandler(this.MYOselectAll_Click);
            // 
            // MYOdisconnect
            // 
            this.MYOdisconnect.Enabled = false;
            this.MYOdisconnect.Location = new System.Drawing.Point(87, 22);
            this.MYOdisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MYOdisconnect.Name = "MYOdisconnect";
            this.MYOdisconnect.Size = new System.Drawing.Size(99, 23);
            this.MYOdisconnect.TabIndex = 210;
            this.MYOdisconnect.Text = "Disconnect";
            this.MYOdisconnect.UseVisualStyleBackColor = true;
            this.MYOdisconnect.Click += new System.EventHandler(this.MYO_disconnect_Click);
            // 
            // MYOlist
            // 
            this.MYOlist.CheckOnClick = true;
            this.MYOlist.Enabled = false;
            this.MYOlist.FormattingEnabled = true;
            this.MYOlist.Items.AddRange(new object[] {
            "Ch1",
            "Ch2",
            "Ch3",
            "Ch4",
            "Ch5",
            "Ch6",
            "Ch7",
            "Ch8"});
            this.MYOlist.Location = new System.Drawing.Point(192, 23);
            this.MYOlist.Margin = new System.Windows.Forms.Padding(4);
            this.MYOlist.Name = "MYOlist";
            this.MYOlist.Size = new System.Drawing.Size(177, 89);
            this.MYOlist.TabIndex = 223;
            // 
            // groupBox8
            // 
            this.groupBox8.Controls.Add(this.pictureBox11);
            this.groupBox8.Controls.Add(this.KBclearAll);
            this.groupBox8.Controls.Add(this.KBlist);
            this.groupBox8.Controls.Add(this.KBselectAll);
            this.groupBox8.Controls.Add(this.KBcheckRamp);
            this.groupBox8.Controls.Add(this.KBlabelRamp);
            this.groupBox8.Controls.Add(this.KBconnect);
            this.groupBox8.Controls.Add(this.KBdisconnect);
            this.groupBox8.Location = new System.Drawing.Point(8, 393);
            this.groupBox8.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox8.Name = "groupBox8";
            this.groupBox8.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox8.Size = new System.Drawing.Size(383, 180);
            this.groupBox8.TabIndex = 212;
            this.groupBox8.TabStop = false;
            this.groupBox8.Text = "Keyboard - Setup";
            // 
            // pictureBox11
            // 
            this.pictureBox11.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox11.Image")));
            this.pictureBox11.Location = new System.Drawing.Point(8, 53);
            this.pictureBox11.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox11.Name = "pictureBox11";
            this.pictureBox11.Size = new System.Drawing.Size(177, 86);
            this.pictureBox11.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox11.TabIndex = 225;
            this.pictureBox11.TabStop = false;
            // 
            // KBclearAll
            // 
            this.KBclearAll.Enabled = false;
            this.KBclearAll.Location = new System.Drawing.Point(287, 144);
            this.KBclearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.KBclearAll.Name = "KBclearAll";
            this.KBclearAll.Size = new System.Drawing.Size(88, 23);
            this.KBclearAll.TabIndex = 224;
            this.KBclearAll.Text = "Clear All";
            this.KBclearAll.UseVisualStyleBackColor = true;
            this.KBclearAll.Click += new System.EventHandler(this.KBclearAll_Click);
            // 
            // KBlist
            // 
            this.KBlist.CheckOnClick = true;
            this.KBlist.Enabled = false;
            this.KBlist.FormattingEnabled = true;
            this.KBlist.Items.AddRange(new object[] {
            "W",
            "A",
            "S",
            "D",
            "O",
            "K",
            "L",
            ";",
            "Up",
            "Down",
            "Left",
            "Right",
            "LeftAlt",
            "RightAlt",
            "Space"});
            this.KBlist.Location = new System.Drawing.Point(193, 23);
            this.KBlist.Margin = new System.Windows.Forms.Padding(4);
            this.KBlist.Name = "KBlist";
            this.KBlist.Size = new System.Drawing.Size(177, 89);
            this.KBlist.TabIndex = 223;
            // 
            // KBselectAll
            // 
            this.KBselectAll.Enabled = false;
            this.KBselectAll.Location = new System.Drawing.Point(193, 144);
            this.KBselectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.KBselectAll.Name = "KBselectAll";
            this.KBselectAll.Size = new System.Drawing.Size(88, 23);
            this.KBselectAll.TabIndex = 223;
            this.KBselectAll.Text = "Select All";
            this.KBselectAll.UseVisualStyleBackColor = true;
            this.KBselectAll.Click += new System.EventHandler(this.KBselectAll_Click);
            // 
            // KBcheckRamp
            // 
            this.KBcheckRamp.AutoSize = true;
            this.KBcheckRamp.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckRamp.Enabled = false;
            this.KBcheckRamp.Location = new System.Drawing.Point(39, 149);
            this.KBcheckRamp.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckRamp.Name = "KBcheckRamp";
            this.KBcheckRamp.Size = new System.Drawing.Size(18, 17);
            this.KBcheckRamp.TabIndex = 211;
            this.KBcheckRamp.UseVisualStyleBackColor = false;
            // 
            // KBlabelRamp
            // 
            this.KBlabelRamp.Enabled = false;
            this.KBlabelRamp.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.KBlabelRamp.Location = new System.Drawing.Point(57, 148);
            this.KBlabelRamp.Name = "KBlabelRamp";
            this.KBlabelRamp.Size = new System.Drawing.Size(111, 18);
            this.KBlabelRamp.TabIndex = 212;
            this.KBlabelRamp.Text = "Velocity Ramp";
            // 
            // KBconnect
            // 
            this.KBconnect.Location = new System.Drawing.Point(8, 22);
            this.KBconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.KBconnect.Name = "KBconnect";
            this.KBconnect.Size = new System.Drawing.Size(75, 23);
            this.KBconnect.TabIndex = 209;
            this.KBconnect.Text = "Connect";
            this.KBconnect.UseVisualStyleBackColor = true;
            this.KBconnect.Click += new System.EventHandler(this.KB_connect_Click);
            // 
            // KBdisconnect
            // 
            this.KBdisconnect.Enabled = false;
            this.KBdisconnect.Location = new System.Drawing.Point(88, 22);
            this.KBdisconnect.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.KBdisconnect.Name = "KBdisconnect";
            this.KBdisconnect.Size = new System.Drawing.Size(99, 23);
            this.KBdisconnect.TabIndex = 210;
            this.KBdisconnect.Text = "Disconnect";
            this.KBdisconnect.UseVisualStyleBackColor = true;
            this.KBdisconnect.Click += new System.EventHandler(this.KB_disconnect_Click);
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.pictureBox9);
            this.groupBox5.Controls.Add(this.XBoxClearAll);
            this.groupBox5.Controls.Add(this.XboxConnect);
            this.groupBox5.Controls.Add(this.XboxDisconnect);
            this.groupBox5.Controls.Add(this.XBoxSelectAll);
            this.groupBox5.Controls.Add(this.XBoxList);
            this.groupBox5.Location = new System.Drawing.Point(8, 18);
            this.groupBox5.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox5.Size = new System.Drawing.Size(383, 180);
            this.groupBox5.TabIndex = 0;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "XBox - Setup";
            // 
            // pictureBox9
            // 
            this.pictureBox9.Image = global::brachIOplexus.Properties.Resources.Xbox_Controller_edit;
            this.pictureBox9.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox9.InitialImage")));
            this.pictureBox9.Location = new System.Drawing.Point(8, 52);
            this.pictureBox9.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox9.Name = "pictureBox9";
            this.pictureBox9.Size = new System.Drawing.Size(177, 116);
            this.pictureBox9.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox9.TabIndex = 222;
            this.pictureBox9.TabStop = false;
            // 
            // XBoxClearAll
            // 
            this.XBoxClearAll.Enabled = false;
            this.XBoxClearAll.Location = new System.Drawing.Point(284, 145);
            this.XBoxClearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.XBoxClearAll.Name = "XBoxClearAll";
            this.XBoxClearAll.Size = new System.Drawing.Size(88, 23);
            this.XBoxClearAll.TabIndex = 221;
            this.XBoxClearAll.Text = "Clear All";
            this.XBoxClearAll.UseVisualStyleBackColor = true;
            this.XBoxClearAll.Click += new System.EventHandler(this.XBoxClearAll_Click);
            // 
            // XBoxSelectAll
            // 
            this.XBoxSelectAll.Enabled = false;
            this.XBoxSelectAll.Location = new System.Drawing.Point(191, 145);
            this.XBoxSelectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.XBoxSelectAll.Name = "XBoxSelectAll";
            this.XBoxSelectAll.Size = new System.Drawing.Size(88, 23);
            this.XBoxSelectAll.TabIndex = 220;
            this.XBoxSelectAll.Text = "Select All";
            this.XBoxSelectAll.UseVisualStyleBackColor = true;
            this.XBoxSelectAll.Click += new System.EventHandler(this.XBoxSelectAll_Click);
            // 
            // XBoxList
            // 
            this.XBoxList.CheckOnClick = true;
            this.XBoxList.Enabled = false;
            this.XBoxList.FormattingEnabled = true;
            this.XBoxList.Items.AddRange(new object[] {
            "StickLeftX1",
            "StickLeftX2",
            "StickLeftY1",
            "StickLeftY2",
            "StickRightX1",
            "StickRightX2",
            "StickRightY1",
            "StickRightY2",
            "TriggerLeft",
            "TriggerRight",
            "StickLeftClick",
            "StickRightClick",
            "ShoulderLeft",
            "ShoulderRight",
            "DPadUp",
            "DPadRight",
            "DPadDown",
            "DPadLeft",
            "A - XBox",
            "B - XBox",
            "X - XBox",
            "Y - XBox",
            "Start",
            "Back",
            "Guide"});
            this.XBoxList.Location = new System.Drawing.Point(193, 23);
            this.XBoxList.Margin = new System.Windows.Forms.Padding(4);
            this.XBoxList.Name = "XBoxList";
            this.XBoxList.Size = new System.Drawing.Size(177, 89);
            this.XBoxList.TabIndex = 219;
            this.XBoxList.ItemCheck += new System.Windows.Forms.ItemCheckEventHandler(this.XBoxList_ItemCheck);
            // 
            // label203
            // 
            this.label203.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label203.Location = new System.Drawing.Point(1351, 315);
            this.label203.Name = "label203";
            this.label203.Size = new System.Drawing.Size(111, 18);
            this.label203.TabIndex = 223;
            this.label203.Text = "Delay (ms):";
            this.label203.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label203.Visible = false;
            // 
            // biopatrecDelay
            // 
            this.biopatrecDelay.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.biopatrecDelay.Location = new System.Drawing.Point(1460, 315);
            this.biopatrecDelay.Name = "biopatrecDelay";
            this.biopatrecDelay.Size = new System.Drawing.Size(56, 18);
            this.biopatrecDelay.TabIndex = 224;
            this.biopatrecDelay.Text = "--";
            this.biopatrecDelay.Visible = false;
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.pictureBox12);
            this.groupBox6.Controls.Add(this.BentoClearAll);
            this.groupBox6.Controls.Add(this.dynaDisconnect);
            this.groupBox6.Controls.Add(this.BentoSelectAll);
            this.groupBox6.Controls.Add(this.cmbSerialPorts);
            this.groupBox6.Controls.Add(this.BentoList);
            this.groupBox6.Controls.Add(this.dynaConnect);
            this.groupBox6.Controls.Add(this.cmbSerialRefresh);
            this.groupBox6.Controls.Add(this.label116);
            this.groupBox6.Location = new System.Drawing.Point(8, 18);
            this.groupBox6.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox6.Size = new System.Drawing.Size(445, 206);
            this.groupBox6.TabIndex = 1;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Bento Arm - Setup";
            // 
            // pictureBox12
            // 
            this.pictureBox12.Image = global::brachIOplexus.Properties.Resources.img_4816_brachIOplexusE;
            this.pictureBox12.InitialImage = ((System.Drawing.Image)(resources.GetObject("pictureBox12.InitialImage")));
            this.pictureBox12.Location = new System.Drawing.Point(8, 49);
            this.pictureBox12.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox12.Name = "pictureBox12";
            this.pictureBox12.Size = new System.Drawing.Size(235, 148);
            this.pictureBox12.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox12.TabIndex = 223;
            this.pictureBox12.TabStop = false;
            // 
            // BentoClearAll
            // 
            this.BentoClearAll.Enabled = false;
            this.BentoClearAll.Location = new System.Drawing.Point(348, 174);
            this.BentoClearAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoClearAll.Name = "BentoClearAll";
            this.BentoClearAll.Size = new System.Drawing.Size(88, 23);
            this.BentoClearAll.TabIndex = 225;
            this.BentoClearAll.Text = "Clear All";
            this.BentoClearAll.UseVisualStyleBackColor = true;
            this.BentoClearAll.Click += new System.EventHandler(this.BentoClearAll_Click);
            // 
            // BentoSelectAll
            // 
            this.BentoSelectAll.Enabled = false;
            this.BentoSelectAll.Location = new System.Drawing.Point(255, 174);
            this.BentoSelectAll.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoSelectAll.Name = "BentoSelectAll";
            this.BentoSelectAll.Size = new System.Drawing.Size(88, 23);
            this.BentoSelectAll.TabIndex = 224;
            this.BentoSelectAll.Text = "Select All";
            this.BentoSelectAll.UseVisualStyleBackColor = true;
            this.BentoSelectAll.Click += new System.EventHandler(this.BentoSelectAll_Click);
            // 
            // BentoList
            // 
            this.BentoList.CheckOnClick = true;
            this.BentoList.Enabled = false;
            this.BentoList.FormattingEnabled = true;
            this.BentoList.Items.AddRange(new object[] {
            "Shoulder (CCW)",
            "Shoulder (CW)",
            "Elbow Extend",
            "Elbow Flex",
            "Wrist (CCW)",
            "Wrist (CW)",
            "Wrist Flex",
            "Wrist Extend",
            "Hand Close",
            "Hand Open",
            "Torque On/Off",
            "Run/Suspend"});
            this.BentoList.Location = new System.Drawing.Point(257, 52);
            this.BentoList.Margin = new System.Windows.Forms.Padding(4);
            this.BentoList.Name = "BentoList";
            this.BentoList.Size = new System.Drawing.Size(177, 89);
            this.BentoList.TabIndex = 223;
            this.BentoList.ItemCheck += new System.Windows.Forms.ItemCheckEventHandler(this.BentoList_ItemCheck);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(1339, 494);
            this.button1.Margin = new System.Windows.Forms.Padding(4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(153, 28);
            this.button1.TabIndex = 214;
            this.button1.Text = "Update Combobox";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Visible = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // comboBox2
            // 
            this.comboBox2.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBox2.FormattingEnabled = true;
            this.comboBox2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.comboBox2.Location = new System.Drawing.Point(1340, 526);
            this.comboBox2.Margin = new System.Windows.Forms.Padding(4);
            this.comboBox2.Name = "comboBox2";
            this.comboBox2.Size = new System.Drawing.Size(160, 24);
            this.comboBox2.TabIndex = 213;
            this.comboBox2.Visible = false;
            // 
            // checkedListFruit
            // 
            this.checkedListFruit.CheckOnClick = true;
            this.checkedListFruit.FormattingEnabled = true;
            this.checkedListFruit.Items.AddRange(new object[] {
            "Apple",
            "Orange",
            "Banana"});
            this.checkedListFruit.Location = new System.Drawing.Point(1375, 348);
            this.checkedListFruit.Margin = new System.Windows.Forms.Padding(4);
            this.checkedListFruit.Name = "checkedListFruit";
            this.checkedListFruit.Size = new System.Drawing.Size(159, 89);
            this.checkedListFruit.TabIndex = 212;
            this.checkedListFruit.Visible = false;
            // 
            // MYOgroupBox
            // 
            this.MYOgroupBox.Controls.Add(this.myo_ch1);
            this.MYOgroupBox.Controls.Add(this.myo_ch2);
            this.MYOgroupBox.Controls.Add(this.label134);
            this.MYOgroupBox.Controls.Add(this.label136);
            this.MYOgroupBox.Controls.Add(this.myo_ch3);
            this.MYOgroupBox.Controls.Add(this.myo_ch4);
            this.MYOgroupBox.Controls.Add(this.myo_ch5);
            this.MYOgroupBox.Controls.Add(this.myo_ch6);
            this.MYOgroupBox.Controls.Add(this.myo_ch7);
            this.MYOgroupBox.Controls.Add(this.label128);
            this.MYOgroupBox.Controls.Add(this.myo_ch8);
            this.MYOgroupBox.Controls.Add(this.label130);
            this.MYOgroupBox.Controls.Add(this.label131);
            this.MYOgroupBox.Controls.Add(this.label133);
            this.MYOgroupBox.Controls.Add(this.label135);
            this.MYOgroupBox.Controls.Add(this.label137);
            this.MYOgroupBox.Enabled = false;
            this.MYOgroupBox.Location = new System.Drawing.Point(4, 234);
            this.MYOgroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.MYOgroupBox.Name = "MYOgroupBox";
            this.MYOgroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.MYOgroupBox.Size = new System.Drawing.Size(349, 202);
            this.MYOgroupBox.TabIndex = 209;
            this.MYOgroupBox.TabStop = false;
            this.MYOgroupBox.Text = "MYO Armband";
            // 
            // myo_ch1
            // 
            this.myo_ch1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch1.Location = new System.Drawing.Point(52, 20);
            this.myo_ch1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch1.Name = "myo_ch1";
            this.myo_ch1.Size = new System.Drawing.Size(79, 19);
            this.myo_ch1.TabIndex = 185;
            this.myo_ch1.Text = "1.0";
            this.myo_ch1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // myo_ch2
            // 
            this.myo_ch2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch2.Location = new System.Drawing.Point(52, 37);
            this.myo_ch2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch2.Name = "myo_ch2";
            this.myo_ch2.Size = new System.Drawing.Size(79, 19);
            this.myo_ch2.TabIndex = 186;
            this.myo_ch2.Text = "1.0";
            this.myo_ch2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label134
            // 
            this.label134.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label134.Location = new System.Drawing.Point(13, 37);
            this.label134.Name = "label134";
            this.label134.Size = new System.Drawing.Size(39, 18);
            this.label134.TabIndex = 188;
            this.label134.Text = "Ch2:";
            this.label134.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label136
            // 
            this.label136.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label136.Location = new System.Drawing.Point(3, 18);
            this.label136.Name = "label136";
            this.label136.Size = new System.Drawing.Size(49, 18);
            this.label136.TabIndex = 187;
            this.label136.Text = "Ch1:";
            this.label136.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // myo_ch3
            // 
            this.myo_ch3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch3.Location = new System.Drawing.Point(52, 64);
            this.myo_ch3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch3.Name = "myo_ch3";
            this.myo_ch3.Size = new System.Drawing.Size(79, 19);
            this.myo_ch3.TabIndex = 157;
            this.myo_ch3.Text = "1.0";
            this.myo_ch3.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // myo_ch4
            // 
            this.myo_ch4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch4.Location = new System.Drawing.Point(52, 81);
            this.myo_ch4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch4.Name = "myo_ch4";
            this.myo_ch4.Size = new System.Drawing.Size(79, 19);
            this.myo_ch4.TabIndex = 158;
            this.myo_ch4.Text = "1.0";
            this.myo_ch4.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // myo_ch5
            // 
            this.myo_ch5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch5.Location = new System.Drawing.Point(52, 107);
            this.myo_ch5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch5.Name = "myo_ch5";
            this.myo_ch5.Size = new System.Drawing.Size(79, 19);
            this.myo_ch5.TabIndex = 159;
            this.myo_ch5.Text = "1.0";
            this.myo_ch5.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // myo_ch6
            // 
            this.myo_ch6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch6.Location = new System.Drawing.Point(52, 126);
            this.myo_ch6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch6.Name = "myo_ch6";
            this.myo_ch6.Size = new System.Drawing.Size(79, 18);
            this.myo_ch6.TabIndex = 160;
            this.myo_ch6.Text = "1.0";
            this.myo_ch6.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // myo_ch7
            // 
            this.myo_ch7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch7.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch7.Location = new System.Drawing.Point(52, 150);
            this.myo_ch7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch7.Name = "myo_ch7";
            this.myo_ch7.Size = new System.Drawing.Size(79, 18);
            this.myo_ch7.TabIndex = 161;
            this.myo_ch7.Text = "1.0";
            this.myo_ch7.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label128
            // 
            this.label128.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label128.Location = new System.Drawing.Point(7, 166);
            this.label128.Name = "label128";
            this.label128.Size = new System.Drawing.Size(44, 18);
            this.label128.TabIndex = 184;
            this.label128.Text = "Ch8:";
            this.label128.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // myo_ch8
            // 
            this.myo_ch8.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.myo_ch8.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.myo_ch8.Location = new System.Drawing.Point(52, 167);
            this.myo_ch8.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.myo_ch8.Name = "myo_ch8";
            this.myo_ch8.Size = new System.Drawing.Size(79, 18);
            this.myo_ch8.TabIndex = 162;
            this.myo_ch8.Text = "1.0";
            this.myo_ch8.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label130
            // 
            this.label130.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label130.Location = new System.Drawing.Point(7, 148);
            this.label130.Name = "label130";
            this.label130.Size = new System.Drawing.Size(44, 18);
            this.label130.TabIndex = 183;
            this.label130.Text = "Ch7:";
            this.label130.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label131
            // 
            this.label131.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label131.Location = new System.Drawing.Point(7, 126);
            this.label131.Name = "label131";
            this.label131.Size = new System.Drawing.Size(44, 18);
            this.label131.TabIndex = 182;
            this.label131.Text = "Ch6:";
            this.label131.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label133
            // 
            this.label133.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label133.Location = new System.Drawing.Point(13, 107);
            this.label133.Name = "label133";
            this.label133.Size = new System.Drawing.Size(39, 18);
            this.label133.TabIndex = 181;
            this.label133.Text = "Ch5:";
            this.label133.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label135
            // 
            this.label135.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label135.Location = new System.Drawing.Point(13, 81);
            this.label135.Name = "label135";
            this.label135.Size = new System.Drawing.Size(39, 18);
            this.label135.TabIndex = 180;
            this.label135.Text = "Ch4:";
            this.label135.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label137
            // 
            this.label137.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label137.Location = new System.Drawing.Point(13, 63);
            this.label137.Name = "label137";
            this.label137.Size = new System.Drawing.Size(39, 18);
            this.label137.TabIndex = 179;
            this.label137.Text = "Ch3:";
            this.label137.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // KBgroupBox
            // 
            this.KBgroupBox.Controls.Add(this.KBrampS);
            this.KBgroupBox.Controls.Add(this.KBrampD);
            this.KBgroupBox.Controls.Add(this.KBrampW);
            this.KBgroupBox.Controls.Add(this.KBcheckRightAlt);
            this.KBgroupBox.Controls.Add(this.KBrampA);
            this.KBgroupBox.Controls.Add(this.KBcheckSpace);
            this.KBgroupBox.Controls.Add(this.KBcheckLeftAlt);
            this.KBgroupBox.Controls.Add(this.label142);
            this.KBgroupBox.Controls.Add(this.label143);
            this.KBgroupBox.Controls.Add(this.label144);
            this.KBgroupBox.Controls.Add(this.KBcheckRight);
            this.KBgroupBox.Controls.Add(this.KBcheckDown);
            this.KBgroupBox.Controls.Add(this.KBcheckLeft);
            this.KBgroupBox.Controls.Add(this.KBcheckUp);
            this.KBgroupBox.Controls.Add(this.label138);
            this.KBgroupBox.Controls.Add(this.label139);
            this.KBgroupBox.Controls.Add(this.label140);
            this.KBgroupBox.Controls.Add(this.label141);
            this.KBgroupBox.Controls.Add(this.KBcheckSemiColon);
            this.KBgroupBox.Controls.Add(this.KBcheckL);
            this.KBgroupBox.Controls.Add(this.KBcheckK);
            this.KBgroupBox.Controls.Add(this.KBcheckO);
            this.KBgroupBox.Controls.Add(this.label126);
            this.KBgroupBox.Controls.Add(this.label127);
            this.KBgroupBox.Controls.Add(this.label129);
            this.KBgroupBox.Controls.Add(this.label132);
            this.KBgroupBox.Controls.Add(this.KBcheckD);
            this.KBgroupBox.Controls.Add(this.KBcheckS);
            this.KBgroupBox.Controls.Add(this.KBcheckA);
            this.KBgroupBox.Controls.Add(this.KBcheckW);
            this.KBgroupBox.Controls.Add(this.label122);
            this.KBgroupBox.Controls.Add(this.label123);
            this.KBgroupBox.Controls.Add(this.label124);
            this.KBgroupBox.Controls.Add(this.label125);
            this.KBgroupBox.Enabled = false;
            this.KBgroupBox.Location = new System.Drawing.Point(4, 443);
            this.KBgroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.KBgroupBox.Name = "KBgroupBox";
            this.KBgroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.KBgroupBox.Size = new System.Drawing.Size(349, 202);
            this.KBgroupBox.TabIndex = 210;
            this.KBgroupBox.TabStop = false;
            this.KBgroupBox.Text = "Keyboard";
            // 
            // KBrampS
            // 
            this.KBrampS.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.KBrampS.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.KBrampS.Location = new System.Drawing.Point(57, 54);
            this.KBrampS.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.KBrampS.Name = "KBrampS";
            this.KBrampS.Size = new System.Drawing.Size(79, 19);
            this.KBrampS.TabIndex = 197;
            this.KBrampS.Text = "0.0";
            this.KBrampS.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // KBrampD
            // 
            this.KBrampD.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.KBrampD.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.KBrampD.Location = new System.Drawing.Point(57, 71);
            this.KBrampD.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.KBrampD.Name = "KBrampD";
            this.KBrampD.Size = new System.Drawing.Size(79, 19);
            this.KBrampD.TabIndex = 198;
            this.KBrampD.Text = "0.0";
            this.KBrampD.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // KBrampW
            // 
            this.KBrampW.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.KBrampW.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.KBrampW.Location = new System.Drawing.Point(57, 21);
            this.KBrampW.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.KBrampW.Name = "KBrampW";
            this.KBrampW.Size = new System.Drawing.Size(79, 19);
            this.KBrampW.TabIndex = 189;
            this.KBrampW.Text = "0.0";
            this.KBrampW.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // KBcheckRightAlt
            // 
            this.KBcheckRightAlt.AutoSize = true;
            this.KBcheckRightAlt.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckRightAlt.Enabled = false;
            this.KBcheckRightAlt.Location = new System.Drawing.Point(205, 137);
            this.KBcheckRightAlt.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckRightAlt.Name = "KBcheckRightAlt";
            this.KBcheckRightAlt.Size = new System.Drawing.Size(18, 17);
            this.KBcheckRightAlt.TabIndex = 191;
            this.KBcheckRightAlt.UseVisualStyleBackColor = false;
            // 
            // KBrampA
            // 
            this.KBrampA.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.KBrampA.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.KBrampA.Location = new System.Drawing.Point(57, 38);
            this.KBrampA.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.KBrampA.Name = "KBrampA";
            this.KBrampA.Size = new System.Drawing.Size(79, 19);
            this.KBrampA.TabIndex = 190;
            this.KBrampA.Text = "0.0";
            this.KBrampA.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // KBcheckSpace
            // 
            this.KBcheckSpace.AutoSize = true;
            this.KBcheckSpace.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckSpace.Enabled = false;
            this.KBcheckSpace.Location = new System.Drawing.Point(205, 118);
            this.KBcheckSpace.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckSpace.Name = "KBcheckSpace";
            this.KBcheckSpace.Size = new System.Drawing.Size(18, 17);
            this.KBcheckSpace.TabIndex = 192;
            this.KBcheckSpace.UseVisualStyleBackColor = false;
            // 
            // KBcheckLeftAlt
            // 
            this.KBcheckLeftAlt.AutoSize = true;
            this.KBcheckLeftAlt.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckLeftAlt.Enabled = false;
            this.KBcheckLeftAlt.Location = new System.Drawing.Point(205, 101);
            this.KBcheckLeftAlt.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckLeftAlt.Name = "KBcheckLeftAlt";
            this.KBcheckLeftAlt.Size = new System.Drawing.Size(18, 17);
            this.KBcheckLeftAlt.TabIndex = 193;
            this.KBcheckLeftAlt.UseVisualStyleBackColor = false;
            // 
            // label142
            // 
            this.label142.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label142.Location = new System.Drawing.Point(147, 100);
            this.label142.Name = "label142";
            this.label142.Size = new System.Drawing.Size(59, 18);
            this.label142.TabIndex = 194;
            this.label142.Text = "LeftAlt:";
            this.label142.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label143
            // 
            this.label143.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label143.Location = new System.Drawing.Point(149, 116);
            this.label143.Name = "label143";
            this.label143.Size = new System.Drawing.Size(56, 18);
            this.label143.TabIndex = 195;
            this.label143.Text = "Space:";
            this.label143.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label144
            // 
            this.label144.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label144.Location = new System.Drawing.Point(140, 135);
            this.label144.Name = "label144";
            this.label144.Size = new System.Drawing.Size(65, 18);
            this.label144.TabIndex = 196;
            this.label144.Text = "RightAlt:";
            this.label144.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // KBcheckRight
            // 
            this.KBcheckRight.AutoSize = true;
            this.KBcheckRight.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckRight.Enabled = false;
            this.KBcheckRight.Location = new System.Drawing.Point(205, 74);
            this.KBcheckRight.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckRight.Name = "KBcheckRight";
            this.KBcheckRight.Size = new System.Drawing.Size(18, 17);
            this.KBcheckRight.TabIndex = 183;
            this.KBcheckRight.UseVisualStyleBackColor = false;
            // 
            // KBcheckDown
            // 
            this.KBcheckDown.AutoSize = true;
            this.KBcheckDown.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckDown.Enabled = false;
            this.KBcheckDown.Location = new System.Drawing.Point(205, 55);
            this.KBcheckDown.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckDown.Name = "KBcheckDown";
            this.KBcheckDown.Size = new System.Drawing.Size(18, 17);
            this.KBcheckDown.TabIndex = 184;
            this.KBcheckDown.UseVisualStyleBackColor = false;
            // 
            // KBcheckLeft
            // 
            this.KBcheckLeft.AutoSize = true;
            this.KBcheckLeft.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckLeft.Enabled = false;
            this.KBcheckLeft.Location = new System.Drawing.Point(205, 38);
            this.KBcheckLeft.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckLeft.Name = "KBcheckLeft";
            this.KBcheckLeft.Size = new System.Drawing.Size(18, 17);
            this.KBcheckLeft.TabIndex = 185;
            this.KBcheckLeft.UseVisualStyleBackColor = false;
            // 
            // KBcheckUp
            // 
            this.KBcheckUp.AutoSize = true;
            this.KBcheckUp.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckUp.Enabled = false;
            this.KBcheckUp.Location = new System.Drawing.Point(205, 21);
            this.KBcheckUp.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckUp.Name = "KBcheckUp";
            this.KBcheckUp.Size = new System.Drawing.Size(18, 17);
            this.KBcheckUp.TabIndex = 186;
            this.KBcheckUp.UseVisualStyleBackColor = false;
            // 
            // label138
            // 
            this.label138.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label138.Location = new System.Drawing.Point(167, 18);
            this.label138.Name = "label138";
            this.label138.Size = new System.Drawing.Size(39, 18);
            this.label138.TabIndex = 187;
            this.label138.Text = "Up:";
            this.label138.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label139
            // 
            this.label139.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label139.Location = new System.Drawing.Point(167, 37);
            this.label139.Name = "label139";
            this.label139.Size = new System.Drawing.Size(39, 18);
            this.label139.TabIndex = 188;
            this.label139.Text = "Left:";
            this.label139.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label140
            // 
            this.label140.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label140.Location = new System.Drawing.Point(149, 53);
            this.label140.Name = "label140";
            this.label140.Size = new System.Drawing.Size(56, 18);
            this.label140.TabIndex = 189;
            this.label140.Text = "Down:";
            this.label140.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label141
            // 
            this.label141.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label141.Location = new System.Drawing.Point(149, 73);
            this.label141.Name = "label141";
            this.label141.Size = new System.Drawing.Size(56, 18);
            this.label141.TabIndex = 190;
            this.label141.Text = "Right:";
            this.label141.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // KBcheckSemiColon
            // 
            this.KBcheckSemiColon.AutoSize = true;
            this.KBcheckSemiColon.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckSemiColon.Enabled = false;
            this.KBcheckSemiColon.Location = new System.Drawing.Point(35, 158);
            this.KBcheckSemiColon.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckSemiColon.Name = "KBcheckSemiColon";
            this.KBcheckSemiColon.Size = new System.Drawing.Size(18, 17);
            this.KBcheckSemiColon.TabIndex = 175;
            this.KBcheckSemiColon.UseVisualStyleBackColor = false;
            // 
            // KBcheckL
            // 
            this.KBcheckL.AutoSize = true;
            this.KBcheckL.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckL.Enabled = false;
            this.KBcheckL.Location = new System.Drawing.Point(35, 139);
            this.KBcheckL.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckL.Name = "KBcheckL";
            this.KBcheckL.Size = new System.Drawing.Size(18, 17);
            this.KBcheckL.TabIndex = 176;
            this.KBcheckL.UseVisualStyleBackColor = false;
            // 
            // KBcheckK
            // 
            this.KBcheckK.AutoSize = true;
            this.KBcheckK.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckK.Enabled = false;
            this.KBcheckK.Location = new System.Drawing.Point(35, 122);
            this.KBcheckK.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckK.Name = "KBcheckK";
            this.KBcheckK.Size = new System.Drawing.Size(18, 17);
            this.KBcheckK.TabIndex = 177;
            this.KBcheckK.UseVisualStyleBackColor = false;
            // 
            // KBcheckO
            // 
            this.KBcheckO.AutoSize = true;
            this.KBcheckO.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckO.Enabled = false;
            this.KBcheckO.Location = new System.Drawing.Point(35, 105);
            this.KBcheckO.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckO.Name = "KBcheckO";
            this.KBcheckO.Size = new System.Drawing.Size(18, 17);
            this.KBcheckO.TabIndex = 178;
            this.KBcheckO.UseVisualStyleBackColor = false;
            // 
            // label126
            // 
            this.label126.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label126.Location = new System.Drawing.Point(7, 102);
            this.label126.Name = "label126";
            this.label126.Size = new System.Drawing.Size(28, 18);
            this.label126.TabIndex = 179;
            this.label126.Text = "O:";
            this.label126.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label127
            // 
            this.label127.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label127.Location = new System.Drawing.Point(7, 121);
            this.label127.Name = "label127";
            this.label127.Size = new System.Drawing.Size(28, 18);
            this.label127.TabIndex = 180;
            this.label127.Text = "K:";
            this.label127.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label129
            // 
            this.label129.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label129.Location = new System.Drawing.Point(7, 137);
            this.label129.Name = "label129";
            this.label129.Size = new System.Drawing.Size(28, 18);
            this.label129.TabIndex = 181;
            this.label129.Text = "L:";
            this.label129.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label132
            // 
            this.label132.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label132.Location = new System.Drawing.Point(7, 156);
            this.label132.Name = "label132";
            this.label132.Size = new System.Drawing.Size(28, 18);
            this.label132.TabIndex = 182;
            this.label132.Text = ";:";
            this.label132.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // KBcheckD
            // 
            this.KBcheckD.AutoSize = true;
            this.KBcheckD.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckD.Enabled = false;
            this.KBcheckD.Location = new System.Drawing.Point(35, 75);
            this.KBcheckD.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckD.Name = "KBcheckD";
            this.KBcheckD.Size = new System.Drawing.Size(18, 17);
            this.KBcheckD.TabIndex = 167;
            this.KBcheckD.UseVisualStyleBackColor = false;
            // 
            // KBcheckS
            // 
            this.KBcheckS.AutoSize = true;
            this.KBcheckS.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckS.Enabled = false;
            this.KBcheckS.Location = new System.Drawing.Point(35, 57);
            this.KBcheckS.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckS.Name = "KBcheckS";
            this.KBcheckS.Size = new System.Drawing.Size(18, 17);
            this.KBcheckS.TabIndex = 168;
            this.KBcheckS.UseVisualStyleBackColor = false;
            // 
            // KBcheckA
            // 
            this.KBcheckA.AutoSize = true;
            this.KBcheckA.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckA.Enabled = false;
            this.KBcheckA.Location = new System.Drawing.Point(35, 39);
            this.KBcheckA.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckA.Name = "KBcheckA";
            this.KBcheckA.Size = new System.Drawing.Size(18, 17);
            this.KBcheckA.TabIndex = 169;
            this.KBcheckA.UseVisualStyleBackColor = false;
            // 
            // KBcheckW
            // 
            this.KBcheckW.AutoSize = true;
            this.KBcheckW.BackColor = System.Drawing.Color.Transparent;
            this.KBcheckW.Enabled = false;
            this.KBcheckW.Location = new System.Drawing.Point(35, 22);
            this.KBcheckW.Margin = new System.Windows.Forms.Padding(4);
            this.KBcheckW.Name = "KBcheckW";
            this.KBcheckW.Size = new System.Drawing.Size(18, 17);
            this.KBcheckW.TabIndex = 170;
            this.KBcheckW.UseVisualStyleBackColor = false;
            // 
            // label122
            // 
            this.label122.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label122.Location = new System.Drawing.Point(7, 20);
            this.label122.Name = "label122";
            this.label122.Size = new System.Drawing.Size(28, 18);
            this.label122.TabIndex = 171;
            this.label122.Text = "W:";
            this.label122.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label123
            // 
            this.label123.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label123.Location = new System.Drawing.Point(7, 38);
            this.label123.Name = "label123";
            this.label123.Size = new System.Drawing.Size(28, 18);
            this.label123.TabIndex = 172;
            this.label123.Text = "A:";
            this.label123.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label124
            // 
            this.label124.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label124.Location = new System.Drawing.Point(7, 54);
            this.label124.Name = "label124";
            this.label124.Size = new System.Drawing.Size(28, 18);
            this.label124.TabIndex = 173;
            this.label124.Text = "S:";
            this.label124.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label125
            // 
            this.label125.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label125.Location = new System.Drawing.Point(7, 74);
            this.label125.Name = "label125";
            this.label125.Size = new System.Drawing.Size(28, 18);
            this.label125.TabIndex = 174;
            this.label125.Text = "D:";
            this.label125.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // tabControl1
            // 
            this.tabControl1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.tabControl1.Controls.Add(this.tabIO);
            this.tabControl1.Controls.Add(this.tabMapping);
            this.tabControl1.Controls.Add(this.tabBento);
            this.tabControl1.Controls.Add(this.tabXPC);
            this.tabControl1.Controls.Add(this.tabViz);
            this.tabControl1.Location = new System.Drawing.Point(11, 33);
            this.tabControl1.Margin = new System.Windows.Forms.Padding(4);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(1560, 822);
            this.tabControl1.TabIndex = 215;
            this.tabControl1.SelectedIndexChanged += new System.EventHandler(this.tabControl1_SelectedIndexChanged);
            this.tabControl1.Deselecting += new System.Windows.Forms.TabControlCancelEventHandler(this.tabControl1_Deselecting);
            // 
            // tabIO
            // 
            this.tabIO.Controls.Add(this.label203);
            this.tabIO.Controls.Add(this.biopatrecDelay);
            this.tabIO.Controls.Add(this.demoShutdownButton);
            this.tabIO.Controls.Add(this.demoMYObutton);
            this.tabIO.Controls.Add(this.demoXBoxButton);
            this.tabIO.Controls.Add(this.InputComboBox);
            this.tabIO.Controls.Add(this.OutputComboBox);
            this.tabIO.Controls.Add(this.label166);
            this.tabIO.Controls.Add(this.labelType);
            this.tabIO.Controls.Add(this.checkedListDairy);
            this.tabIO.Controls.Add(this.button14);
            this.tabIO.Controls.Add(this.labelID);
            this.tabIO.Controls.Add(this.labelText);
            this.tabIO.Controls.Add(this.groupBox9);
            this.tabIO.Controls.Add(this.groupBox4);
            this.tabIO.Controls.Add(this.button1);
            this.tabIO.Controls.Add(this.checkedListFruit);
            this.tabIO.Controls.Add(this.comboBox2);
            this.tabIO.Controls.Add(this.label121);
            this.tabIO.Controls.Add(this.delay_max);
            this.tabIO.Controls.Add(this.dynaStatus);
            this.tabIO.Controls.Add(this.label119);
            this.tabIO.Location = new System.Drawing.Point(4, 25);
            this.tabIO.Margin = new System.Windows.Forms.Padding(4);
            this.tabIO.Name = "tabIO";
            this.tabIO.Padding = new System.Windows.Forms.Padding(4);
            this.tabIO.Size = new System.Drawing.Size(1552, 793);
            this.tabIO.TabIndex = 0;
            this.tabIO.Text = "Input/Output";
            this.tabIO.UseVisualStyleBackColor = true;
            // 
            // demoShutdownButton
            // 
            this.demoShutdownButton.Location = new System.Drawing.Point(969, 453);
            this.demoShutdownButton.Margin = new System.Windows.Forms.Padding(4);
            this.demoShutdownButton.Name = "demoShutdownButton";
            this.demoShutdownButton.Size = new System.Drawing.Size(137, 59);
            this.demoShutdownButton.TabIndex = 228;
            this.demoShutdownButton.Text = "Shutdown Demos";
            this.demoShutdownButton.UseVisualStyleBackColor = true;
            this.demoShutdownButton.Click += new System.EventHandler(this.demoShutdownButton_Click);
            // 
            // demoMYObutton
            // 
            this.demoMYObutton.Location = new System.Drawing.Point(969, 385);
            this.demoMYObutton.Margin = new System.Windows.Forms.Padding(4);
            this.demoMYObutton.Name = "demoMYObutton";
            this.demoMYObutton.Size = new System.Drawing.Size(137, 59);
            this.demoMYObutton.TabIndex = 227;
            this.demoMYObutton.Text = "Start MYO Demo";
            this.demoMYObutton.UseVisualStyleBackColor = true;
            this.demoMYObutton.Click += new System.EventHandler(this.demoMYObutton_Click);
            // 
            // demoXBoxButton
            // 
            this.demoXBoxButton.Location = new System.Drawing.Point(969, 321);
            this.demoXBoxButton.Margin = new System.Windows.Forms.Padding(4);
            this.demoXBoxButton.Name = "demoXBoxButton";
            this.demoXBoxButton.Size = new System.Drawing.Size(137, 59);
            this.demoXBoxButton.TabIndex = 226;
            this.demoXBoxButton.Text = "Start XBox Demo";
            this.demoXBoxButton.UseVisualStyleBackColor = true;
            this.demoXBoxButton.Click += new System.EventHandler(this.demoXBoxButton_Click);
            // 
            // InputComboBox
            // 
            this.InputComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.InputComboBox.FormattingEnabled = true;
            this.InputComboBox.Location = new System.Drawing.Point(1223, 636);
            this.InputComboBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.InputComboBox.Name = "InputComboBox";
            this.InputComboBox.Size = new System.Drawing.Size(131, 24);
            this.InputComboBox.TabIndex = 222;
            this.InputComboBox.Visible = false;
            // 
            // OutputComboBox
            // 
            this.OutputComboBox.DisplayMember = "1";
            this.OutputComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.OutputComboBox.FormattingEnabled = true;
            this.OutputComboBox.Location = new System.Drawing.Point(1384, 636);
            this.OutputComboBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.OutputComboBox.Name = "OutputComboBox";
            this.OutputComboBox.Size = new System.Drawing.Size(131, 24);
            this.OutputComboBox.TabIndex = 221;
            this.OutputComboBox.Visible = false;
            // 
            // label166
            // 
            this.label166.AutoSize = true;
            this.label166.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label166.Location = new System.Drawing.Point(1356, 640);
            this.label166.Name = "label166";
            this.label166.Size = new System.Drawing.Size(24, 17);
            this.label166.TabIndex = 220;
            this.label166.Text = ">>";
            this.label166.Visible = false;
            // 
            // labelType
            // 
            this.labelType.AutoSize = true;
            this.labelType.Location = new System.Drawing.Point(1345, 591);
            this.labelType.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelType.Name = "labelType";
            this.labelType.Size = new System.Drawing.Size(42, 17);
            this.labelType.TabIndex = 219;
            this.labelType.Text = "value";
            this.labelType.Visible = false;
            // 
            // checkedListDairy
            // 
            this.checkedListDairy.CheckOnClick = true;
            this.checkedListDairy.FormattingEnabled = true;
            this.checkedListDairy.Items.AddRange(new object[] {
            "Cheese",
            "Milk"});
            this.checkedListDairy.Location = new System.Drawing.Point(1207, 348);
            this.checkedListDairy.Margin = new System.Windows.Forms.Padding(4);
            this.checkedListDairy.Name = "checkedListDairy";
            this.checkedListDairy.Size = new System.Drawing.Size(159, 89);
            this.checkedListDairy.TabIndex = 218;
            this.checkedListDairy.Visible = false;
            // 
            // button14
            // 
            this.button14.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.button14.Location = new System.Drawing.Point(1340, 459);
            this.button14.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.button14.Name = "button14";
            this.button14.Size = new System.Drawing.Size(145, 28);
            this.button14.TabIndex = 12;
            this.button14.Text = "Update Text";
            this.button14.Visible = false;
            this.button14.Click += new System.EventHandler(this.button14_Click);
            // 
            // labelID
            // 
            this.labelID.AutoSize = true;
            this.labelID.Location = new System.Drawing.Point(1345, 575);
            this.labelID.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelID.Name = "labelID";
            this.labelID.Size = new System.Drawing.Size(21, 17);
            this.labelID.TabIndex = 217;
            this.labelID.Text = "ID";
            this.labelID.Visible = false;
            // 
            // labelText
            // 
            this.labelText.AutoSize = true;
            this.labelText.Location = new System.Drawing.Point(1345, 607);
            this.labelText.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.labelText.Name = "labelText";
            this.labelText.Size = new System.Drawing.Size(30, 17);
            this.labelText.TabIndex = 216;
            this.labelText.Text = "text";
            this.labelText.Visible = false;
            // 
            // groupBox9
            // 
            this.groupBox9.Controls.Add(this.groupBox6);
            this.groupBox9.Location = new System.Drawing.Point(803, 7);
            this.groupBox9.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox9.Name = "groupBox9";
            this.groupBox9.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox9.Size = new System.Drawing.Size(464, 234);
            this.groupBox9.TabIndex = 215;
            this.groupBox9.TabStop = false;
            this.groupBox9.Text = "Output Devices";
            // 
            // tabMapping
            // 
            this.tabMapping.Controls.Add(this.LoggingGroupBox);
            this.tabMapping.Controls.Add(this.groupBox16);
            this.tabMapping.Controls.Add(this.label162);
            this.tabMapping.Controls.Add(this.label150);
            this.tabMapping.Controls.Add(this.label146);
            this.tabMapping.Controls.Add(this.label163);
            this.tabMapping.Controls.Add(this.label158);
            this.tabMapping.Controls.Add(this.label157);
            this.tabMapping.Controls.Add(this.label156);
            this.tabMapping.Controls.Add(this.doF6);
            this.tabMapping.Controls.Add(this.doF5);
            this.tabMapping.Controls.Add(this.doF4);
            this.tabMapping.Controls.Add(this.doF3);
            this.tabMapping.Controls.Add(this.doF2);
            this.tabMapping.Controls.Add(this.doF1);
            this.tabMapping.Location = new System.Drawing.Point(4, 25);
            this.tabMapping.Margin = new System.Windows.Forms.Padding(4);
            this.tabMapping.Name = "tabMapping";
            this.tabMapping.Padding = new System.Windows.Forms.Padding(4);
            this.tabMapping.Size = new System.Drawing.Size(1552, 793);
            this.tabMapping.TabIndex = 1;
            this.tabMapping.Text = "Mapping";
            this.tabMapping.UseVisualStyleBackColor = true;
            // 
            // LoggingGroupBox
            // 
            this.LoggingGroupBox.Controls.Add(this.label234);
            this.LoggingGroupBox.Controls.Add(this.label233);
            this.LoggingGroupBox.Controls.Add(this.label232);
            this.LoggingGroupBox.Controls.Add(this.label231);
            this.LoggingGroupBox.Controls.Add(this.label230);
            this.LoggingGroupBox.Controls.Add(this.intervention);
            this.LoggingGroupBox.Controls.Add(this.label229);
            this.LoggingGroupBox.Controls.Add(this.task_type);
            this.LoggingGroupBox.Controls.Add(this.label227);
            this.LoggingGroupBox.Controls.Add(this.ppt_no);
            this.LoggingGroupBox.Controls.Add(this.label_ppt_no);
            this.LoggingGroupBox.Controls.Add(this.StartLogging);
            this.LoggingGroupBox.Controls.Add(this.label228);
            this.LoggingGroupBox.Controls.Add(this.StopLogging);
            this.LoggingGroupBox.Controls.Add(this.log_number);
            this.LoggingGroupBox.Enabled = false;
            this.LoggingGroupBox.Location = new System.Drawing.Point(965, 549);
            this.LoggingGroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.LoggingGroupBox.Name = "LoggingGroupBox";
            this.LoggingGroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.LoggingGroupBox.Size = new System.Drawing.Size(496, 197);
            this.LoggingGroupBox.TabIndex = 235;
            this.LoggingGroupBox.TabStop = false;
            this.LoggingGroupBox.Text = "Data Logging";
            // 
            // label234
            // 
            this.label234.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label234.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label234.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label234.Location = new System.Drawing.Point(365, 106);
            this.label234.Name = "label234";
            this.label234.Size = new System.Drawing.Size(124, 21);
            this.label234.TabIndex = 245;
            this.label234.Text = "F/SS/AL";
            // 
            // label233
            // 
            this.label233.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label233.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label233.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label233.Location = new System.Drawing.Point(365, 79);
            this.label233.Name = "label233";
            this.label233.Size = new System.Drawing.Size(124, 21);
            this.label233.TabIndex = 244;
            this.label233.Text = "Pasta/Cups";
            // 
            // label232
            // 
            this.label232.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label232.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label232.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label232.Location = new System.Drawing.Point(365, 50);
            this.label232.Name = "label232";
            this.label232.Size = new System.Drawing.Size(124, 21);
            this.label232.TabIndex = 243;
            this.label232.Text = "Random ID#";
            // 
            // label231
            // 
            this.label231.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label231.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label231.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label231.Location = new System.Drawing.Point(321, 26);
            this.label231.Name = "label231";
            this.label231.Size = new System.Drawing.Size(167, 20);
            this.label231.TabIndex = 242;
            this.label231.Text = "Reset each intervention";
            // 
            // label230
            // 
            this.label230.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label230.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label230.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label230.Location = new System.Drawing.Point(7, 139);
            this.label230.Name = "label230";
            this.label230.Size = new System.Drawing.Size(481, 54);
            this.label230.TabIndex = 241;
            this.label230.Text = "Check filepath for saving in the initialization region of the code before beginni" +
    "ng trials. Creates a file named Pro00077839-03-18-1xx_TaskType_Intervention_LogN" +
    "umber";
            // 
            // intervention
            // 
            this.intervention.Location = new System.Drawing.Point(259, 105);
            this.intervention.Name = "intervention";
            this.intervention.Size = new System.Drawing.Size(100, 22);
            this.intervention.TabIndex = 240;
            // 
            // label229
            // 
            this.label229.AutoSize = true;
            this.label229.Location = new System.Drawing.Point(170, 108);
            this.label229.Name = "label229";
            this.label229.Size = new System.Drawing.Size(82, 17);
            this.label229.TabIndex = 239;
            this.label229.Text = "Intervention";
            // 
            // task_type
            // 
            this.task_type.Location = new System.Drawing.Point(259, 77);
            this.task_type.Name = "task_type";
            this.task_type.Size = new System.Drawing.Size(100, 22);
            this.task_type.TabIndex = 238;
            // 
            // label227
            // 
            this.label227.AutoSize = true;
            this.label227.Location = new System.Drawing.Point(177, 80);
            this.label227.Name = "label227";
            this.label227.Size = new System.Drawing.Size(75, 17);
            this.label227.TabIndex = 237;
            this.label227.Text = "Task Type";
            // 
            // ppt_no
            // 
            this.ppt_no.Location = new System.Drawing.Point(259, 49);
            this.ppt_no.Name = "ppt_no";
            this.ppt_no.Size = new System.Drawing.Size(100, 22);
            this.ppt_no.TabIndex = 236;
            // 
            // label_ppt_no
            // 
            this.label_ppt_no.AutoSize = true;
            this.label_ppt_no.Location = new System.Drawing.Point(124, 51);
            this.label_ppt_no.Name = "label_ppt_no";
            this.label_ppt_no.Size = new System.Drawing.Size(129, 17);
            this.label_ppt_no.TabIndex = 235;
            this.label_ppt_no.Text = "Participant Number";
            // 
            // StartLogging
            // 
            this.StartLogging.Location = new System.Drawing.Point(7, 21);
            this.StartLogging.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.StartLogging.Name = "StartLogging";
            this.StartLogging.Size = new System.Drawing.Size(105, 29);
            this.StartLogging.TabIndex = 187;
            this.StartLogging.Text = "START";
            this.StartLogging.UseVisualStyleBackColor = true;
            this.StartLogging.Click += new System.EventHandler(this.StartLogging_Click_1);
            // 
            // label228
            // 
            this.label228.AutoSize = true;
            this.label228.Location = new System.Drawing.Point(170, 21);
            this.label228.Name = "label228";
            this.label228.Size = new System.Drawing.Size(83, 17);
            this.label228.TabIndex = 234;
            this.label228.Text = "log number:";
            // 
            // StopLogging
            // 
            this.StopLogging.Enabled = false;
            this.StopLogging.Location = new System.Drawing.Point(7, 57);
            this.StopLogging.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.StopLogging.Name = "StopLogging";
            this.StopLogging.Size = new System.Drawing.Size(105, 27);
            this.StopLogging.TabIndex = 188;
            this.StopLogging.Text = "STOP";
            this.StopLogging.UseVisualStyleBackColor = true;
            this.StopLogging.Click += new System.EventHandler(this.StopLogging_Click_1);
            // 
            // log_number
            // 
            this.log_number.Location = new System.Drawing.Point(259, 21);
            this.log_number.Name = "log_number";
            this.log_number.Size = new System.Drawing.Size(61, 22);
            this.log_number.TabIndex = 233;
            this.log_number.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // groupBox16
            // 
            this.groupBox16.Controls.Add(this.switchSmaxLabel2);
            this.groupBox16.Controls.Add(this.switchSminLabel2);
            this.groupBox16.Controls.Add(this.switchSminTick2);
            this.groupBox16.Controls.Add(this.switchState_label);
            this.groupBox16.Controls.Add(this.switchSmaxTick2);
            this.groupBox16.Controls.Add(this.label213);
            this.groupBox16.Controls.Add(this.switchSmaxCtrl2);
            this.groupBox16.Controls.Add(this.flag2_label);
            this.groupBox16.Controls.Add(this.switchSminCtrl2);
            this.groupBox16.Controls.Add(this.label211);
            this.groupBox16.Controls.Add(this.switchSignalBar2);
            this.groupBox16.Controls.Add(this.flag1_label);
            this.groupBox16.Controls.Add(this.switchGainCtrl2);
            this.groupBox16.Controls.Add(this.label209);
            this.groupBox16.Controls.Add(this.switchTimeCtrl2);
            this.groupBox16.Controls.Add(this.timer1_label);
            this.groupBox16.Controls.Add(this.groupBox11);
            this.groupBox16.Controls.Add(this.label205);
            this.groupBox16.Controls.Add(this.ID2_state);
            this.groupBox16.Controls.Add(this.groupBox10);
            this.groupBox16.Controls.Add(this.label148);
            this.groupBox16.Controls.Add(this.label103);
            this.groupBox16.Controls.Add(this.label104);
            this.groupBox16.Controls.Add(this.label145);
            this.groupBox16.Controls.Add(this.label147);
            this.groupBox16.Controls.Add(this.switchSmaxLabel1);
            this.groupBox16.Controls.Add(this.switchSminLabel1);
            this.groupBox16.Controls.Add(this.switchSminTick1);
            this.groupBox16.Controls.Add(this.switchSmaxTick1);
            this.groupBox16.Controls.Add(this.switchSmaxCtrl1);
            this.groupBox16.Controls.Add(this.switchSminCtrl1);
            this.groupBox16.Controls.Add(this.switchSignalBar1);
            this.groupBox16.Controls.Add(this.switchGainCtrl1);
            this.groupBox16.Controls.Add(this.switchInputBox);
            this.groupBox16.Controls.Add(this.label39);
            this.groupBox16.Controls.Add(this.label27);
            this.groupBox16.Controls.Add(this.switchTimeCtrl1);
            this.groupBox16.Controls.Add(this.label242);
            this.groupBox16.Controls.Add(this.switchModeBox);
            this.groupBox16.Controls.Add(this.switchLabel);
            this.groupBox16.Controls.Add(this.label257);
            this.groupBox16.Controls.Add(this.switchDoFbox);
            this.groupBox16.Controls.Add(this.label258);
            this.groupBox16.Location = new System.Drawing.Point(965, 22);
            this.groupBox16.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox16.Name = "groupBox16";
            this.groupBox16.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox16.Size = new System.Drawing.Size(496, 521);
            this.groupBox16.TabIndex = 138;
            this.groupBox16.TabStop = false;
            this.groupBox16.Text = "Sequential Switch";
            // 
            // switchSmaxLabel2
            // 
            this.switchSmaxLabel2.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.switchSmaxLabel2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSmaxLabel2.Location = new System.Drawing.Point(179, 176);
            this.switchSmaxLabel2.Name = "switchSmaxLabel2";
            this.switchSmaxLabel2.Size = new System.Drawing.Size(40, 18);
            this.switchSmaxLabel2.TabIndex = 212;
            this.switchSmaxLabel2.Text = "Smax";
            this.switchSmaxLabel2.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // switchSminLabel2
            // 
            this.switchSminLabel2.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.switchSminLabel2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSminLabel2.Location = new System.Drawing.Point(4, 176);
            this.switchSminLabel2.Name = "switchSminLabel2";
            this.switchSminLabel2.Size = new System.Drawing.Size(35, 18);
            this.switchSminLabel2.TabIndex = 211;
            this.switchSminLabel2.Text = "Smin";
            this.switchSminLabel2.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // switchSminTick2
            // 
            this.switchSminTick2.BackColor = System.Drawing.Color.MediumPurple;
            this.switchSminTick2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSminTick2.Location = new System.Drawing.Point(20, 146);
            this.switchSminTick2.Name = "switchSminTick2";
            this.switchSminTick2.Size = new System.Drawing.Size(3, 30);
            this.switchSminTick2.TabIndex = 210;
            // 
            // switchState_label
            // 
            this.switchState_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchState_label.Location = new System.Drawing.Point(368, 490);
            this.switchState_label.Name = "switchState_label";
            this.switchState_label.Size = new System.Drawing.Size(107, 18);
            this.switchState_label.TabIndex = 232;
            this.switchState_label.Text = "switchState";
            // 
            // switchSmaxTick2
            // 
            this.switchSmaxTick2.BackColor = System.Drawing.Color.MediumPurple;
            this.switchSmaxTick2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSmaxTick2.Location = new System.Drawing.Point(197, 146);
            this.switchSmaxTick2.Name = "switchSmaxTick2";
            this.switchSmaxTick2.Size = new System.Drawing.Size(3, 30);
            this.switchSmaxTick2.TabIndex = 209;
            // 
            // label213
            // 
            this.label213.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label213.Location = new System.Drawing.Point(256, 490);
            this.label213.Name = "label213";
            this.label213.Size = new System.Drawing.Size(107, 18);
            this.label213.TabIndex = 231;
            this.label213.Text = "switchState:";
            this.label213.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // switchSmaxCtrl2
            // 
            this.switchSmaxCtrl2.DecimalPlaces = 1;
            this.switchSmaxCtrl2.Increment = new decimal(new int[] {
            2,
            0,
            0,
            65536});
            this.switchSmaxCtrl2.Location = new System.Drawing.Point(336, 146);
            this.switchSmaxCtrl2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSmaxCtrl2.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSmaxCtrl2.Name = "switchSmaxCtrl2";
            this.switchSmaxCtrl2.Size = new System.Drawing.Size(53, 22);
            this.switchSmaxCtrl2.TabIndex = 208;
            this.switchSmaxCtrl2.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSmaxCtrl2.ValueChanged += new System.EventHandler(this.switchSmaxCtrl2_ValueChanged);
            // 
            // flag2_label
            // 
            this.flag2_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.flag2_label.Location = new System.Drawing.Point(368, 471);
            this.flag2_label.Name = "flag2_label";
            this.flag2_label.Size = new System.Drawing.Size(107, 18);
            this.flag2_label.TabIndex = 230;
            this.flag2_label.Text = "flag2";
            // 
            // switchSminCtrl2
            // 
            this.switchSminCtrl2.DecimalPlaces = 1;
            this.switchSminCtrl2.Increment = new decimal(new int[] {
            2,
            0,
            0,
            65536});
            this.switchSminCtrl2.Location = new System.Drawing.Point(279, 146);
            this.switchSminCtrl2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSminCtrl2.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSminCtrl2.Name = "switchSminCtrl2";
            this.switchSminCtrl2.Size = new System.Drawing.Size(53, 22);
            this.switchSminCtrl2.TabIndex = 207;
            this.switchSminCtrl2.Value = new decimal(new int[] {
            6,
            0,
            0,
            65536});
            this.switchSminCtrl2.ValueChanged += new System.EventHandler(this.switchSminCtrl2_ValueChanged);
            // 
            // label211
            // 
            this.label211.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label211.Location = new System.Drawing.Point(256, 471);
            this.label211.Name = "label211";
            this.label211.Size = new System.Drawing.Size(107, 18);
            this.label211.TabIndex = 229;
            this.label211.Text = "flag2:";
            this.label211.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // switchSignalBar2
            // 
            this.switchSignalBar2.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSignalBar2.Location = new System.Drawing.Point(20, 146);
            this.switchSignalBar2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSignalBar2.MarqueeAnimationSpeed = 30;
            this.switchSignalBar2.Maximum = 500;
            this.switchSignalBar2.Name = "switchSignalBar2";
            this.switchSignalBar2.Size = new System.Drawing.Size(179, 27);
            this.switchSignalBar2.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.switchSignalBar2.TabIndex = 206;
            // 
            // flag1_label
            // 
            this.flag1_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.flag1_label.Location = new System.Drawing.Point(368, 453);
            this.flag1_label.Name = "flag1_label";
            this.flag1_label.Size = new System.Drawing.Size(107, 18);
            this.flag1_label.TabIndex = 228;
            this.flag1_label.Text = "flag1";
            // 
            // switchGainCtrl2
            // 
            this.switchGainCtrl2.Enabled = false;
            this.switchGainCtrl2.Location = new System.Drawing.Point(216, 146);
            this.switchGainCtrl2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchGainCtrl2.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.switchGainCtrl2.Name = "switchGainCtrl2";
            this.switchGainCtrl2.Size = new System.Drawing.Size(56, 22);
            this.switchGainCtrl2.TabIndex = 205;
            this.switchGainCtrl2.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // label209
            // 
            this.label209.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label209.Location = new System.Drawing.Point(256, 453);
            this.label209.Name = "label209";
            this.label209.Size = new System.Drawing.Size(107, 18);
            this.label209.TabIndex = 227;
            this.label209.Text = "flag1:";
            this.label209.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // switchTimeCtrl2
            // 
            this.switchTimeCtrl2.Enabled = false;
            this.switchTimeCtrl2.Location = new System.Drawing.Point(397, 146);
            this.switchTimeCtrl2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchTimeCtrl2.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.switchTimeCtrl2.Name = "switchTimeCtrl2";
            this.switchTimeCtrl2.Size = new System.Drawing.Size(61, 22);
            this.switchTimeCtrl2.TabIndex = 204;
            // 
            // timer1_label
            // 
            this.timer1_label.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.timer1_label.Location = new System.Drawing.Point(368, 435);
            this.timer1_label.Name = "timer1_label";
            this.timer1_label.Size = new System.Drawing.Size(107, 18);
            this.timer1_label.TabIndex = 226;
            this.timer1_label.Text = "timer1";
            // 
            // groupBox11
            // 
            this.groupBox11.Controls.Add(this.groupBox14);
            this.groupBox11.Controls.Add(this.groupBox13);
            this.groupBox11.Controls.Add(this.groupBox12);
            this.groupBox11.Location = new System.Drawing.Point(325, 198);
            this.groupBox11.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox11.Name = "groupBox11";
            this.groupBox11.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox11.Size = new System.Drawing.Size(163, 194);
            this.groupBox11.TabIndex = 203;
            this.groupBox11.TabStop = false;
            this.groupBox11.Text = "Feedback";
            // 
            // groupBox14
            // 
            this.groupBox14.Controls.Add(this.textBox);
            this.groupBox14.Location = new System.Drawing.Point(7, 129);
            this.groupBox14.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox14.Name = "groupBox14";
            this.groupBox14.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox14.Size = new System.Drawing.Size(149, 47);
            this.groupBox14.TabIndex = 208;
            this.groupBox14.TabStop = false;
            this.groupBox14.Text = "Visual";
            // 
            // textBox
            // 
            this.textBox.AutoSize = true;
            this.textBox.Location = new System.Drawing.Point(8, 20);
            this.textBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.textBox.Name = "textBox";
            this.textBox.Size = new System.Drawing.Size(57, 21);
            this.textBox.TabIndex = 173;
            this.textBox.Text = "Text";
            this.textBox.UseVisualStyleBackColor = true;
            // 
            // groupBox13
            // 
            this.groupBox13.Controls.Add(this.XboxBuzzBox);
            this.groupBox13.Controls.Add(this.myoBuzzBox);
            this.groupBox13.Location = new System.Drawing.Point(7, 75);
            this.groupBox13.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox13.Name = "groupBox13";
            this.groupBox13.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox13.Size = new System.Drawing.Size(149, 47);
            this.groupBox13.TabIndex = 207;
            this.groupBox13.TabStop = false;
            this.groupBox13.Text = "Vibration";
            // 
            // XboxBuzzBox
            // 
            this.XboxBuzzBox.AutoSize = true;
            this.XboxBuzzBox.Location = new System.Drawing.Point(8, 20);
            this.XboxBuzzBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.XboxBuzzBox.Name = "XboxBuzzBox";
            this.XboxBuzzBox.Size = new System.Drawing.Size(61, 21);
            this.XboxBuzzBox.TabIndex = 175;
            this.XboxBuzzBox.Text = "Xbox";
            this.XboxBuzzBox.UseVisualStyleBackColor = true;
            // 
            // myoBuzzBox
            // 
            this.myoBuzzBox.AutoSize = true;
            this.myoBuzzBox.Location = new System.Drawing.Point(75, 20);
            this.myoBuzzBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.myoBuzzBox.Name = "myoBuzzBox";
            this.myoBuzzBox.Size = new System.Drawing.Size(61, 21);
            this.myoBuzzBox.TabIndex = 174;
            this.myoBuzzBox.Text = "MYO";
            this.myoBuzzBox.UseVisualStyleBackColor = true;
            // 
            // groupBox12
            // 
            this.groupBox12.Controls.Add(this.dingBox);
            this.groupBox12.Controls.Add(this.vocalBox);
            this.groupBox12.Location = new System.Drawing.Point(7, 21);
            this.groupBox12.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox12.Name = "groupBox12";
            this.groupBox12.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox12.Size = new System.Drawing.Size(149, 47);
            this.groupBox12.TabIndex = 206;
            this.groupBox12.TabStop = false;
            this.groupBox12.Text = "Auditory";
            // 
            // dingBox
            // 
            this.dingBox.AutoSize = true;
            this.dingBox.Location = new System.Drawing.Point(8, 20);
            this.dingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.dingBox.Name = "dingBox";
            this.dingBox.Size = new System.Drawing.Size(59, 21);
            this.dingBox.TabIndex = 157;
            this.dingBox.Text = "Ding";
            this.dingBox.UseVisualStyleBackColor = true;
            // 
            // vocalBox
            // 
            this.vocalBox.AccessibleDescription = "";
            this.vocalBox.AutoSize = true;
            this.vocalBox.Location = new System.Drawing.Point(75, 20);
            this.vocalBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.vocalBox.Name = "vocalBox";
            this.vocalBox.Size = new System.Drawing.Size(65, 21);
            this.vocalBox.TabIndex = 158;
            this.vocalBox.Text = "Vocal";
            this.vocalBox.UseVisualStyleBackColor = true;
            // 
            // label205
            // 
            this.label205.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label205.Location = new System.Drawing.Point(256, 435);
            this.label205.Name = "label205";
            this.label205.Size = new System.Drawing.Size(107, 18);
            this.label205.TabIndex = 225;
            this.label205.Text = "timer1:";
            this.label205.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // ID2_state
            // 
            this.ID2_state.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.ID2_state.Location = new System.Drawing.Point(473, 539);
            this.ID2_state.Name = "ID2_state";
            this.ID2_state.Size = new System.Drawing.Size(107, 18);
            this.ID2_state.TabIndex = 224;
            this.ID2_state.Text = "ID2 State:";
            this.ID2_state.Visible = false;
            // 
            // groupBox10
            // 
            this.groupBox10.Controls.Add(this.label37);
            this.groupBox10.Controls.Add(this.switch1Flip);
            this.groupBox10.Controls.Add(this.switch1OutputBox);
            this.groupBox10.Controls.Add(this.label238);
            this.groupBox10.Controls.Add(this.switch2Flip);
            this.groupBox10.Controls.Add(this.switch5MappingBox);
            this.groupBox10.Controls.Add(this.switch4MappingBox);
            this.groupBox10.Controls.Add(this.switch2OutputBox);
            this.groupBox10.Controls.Add(this.switch3Flip);
            this.groupBox10.Controls.Add(this.label240);
            this.groupBox10.Controls.Add(this.label253);
            this.groupBox10.Controls.Add(this.label239);
            this.groupBox10.Controls.Add(this.label38);
            this.groupBox10.Controls.Add(this.switch5Flip);
            this.groupBox10.Controls.Add(this.label241);
            this.groupBox10.Controls.Add(this.label237);
            this.groupBox10.Controls.Add(this.switch1MappingBox);
            this.groupBox10.Controls.Add(this.switch3OutputBox);
            this.groupBox10.Controls.Add(this.switch4Flip);
            this.groupBox10.Controls.Add(this.switch3MappingBox);
            this.groupBox10.Controls.Add(this.switch5OutputBox);
            this.groupBox10.Controls.Add(this.switch2MappingBox);
            this.groupBox10.Controls.Add(this.switch4OutputBox);
            this.groupBox10.Location = new System.Drawing.Point(7, 198);
            this.groupBox10.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox10.Name = "groupBox10";
            this.groupBox10.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox10.Size = new System.Drawing.Size(313, 194);
            this.groupBox10.TabIndex = 202;
            this.groupBox10.TabStop = false;
            this.groupBox10.Text = "Switching List";
            // 
            // label37
            // 
            this.label37.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label37.Location = new System.Drawing.Point(5, 26);
            this.label37.Name = "label37";
            this.label37.Size = new System.Drawing.Size(108, 17);
            this.label37.TabIndex = 185;
            this.label37.Text = "Output Device:";
            // 
            // switch1Flip
            // 
            this.switch1Flip.AutoSize = true;
            this.switch1Flip.Location = new System.Drawing.Point(277, 59);
            this.switch1Flip.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch1Flip.Name = "switch1Flip";
            this.switch1Flip.Size = new System.Drawing.Size(18, 17);
            this.switch1Flip.TabIndex = 151;
            this.switch1Flip.UseVisualStyleBackColor = true;
            this.switch1Flip.CheckedChanged += new System.EventHandler(this.switch1Flip_CheckedChanged);
            // 
            // switch1OutputBox
            // 
            this.switch1OutputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch1OutputBox.FormattingEnabled = true;
            this.switch1OutputBox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.switch1OutputBox.Location = new System.Drawing.Point(9, 55);
            this.switch1OutputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch1OutputBox.Name = "switch1OutputBox";
            this.switch1OutputBox.Size = new System.Drawing.Size(111, 24);
            this.switch1OutputBox.TabIndex = 53;
            this.switch1OutputBox.SelectedIndexChanged += new System.EventHandler(this.switch1OutputBox_SelectedIndexChanged);
            // 
            // label238
            // 
            this.label238.AutoSize = true;
            this.label238.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label238.Location = new System.Drawing.Point(127, 138);
            this.label238.Name = "label238";
            this.label238.Size = new System.Drawing.Size(23, 17);
            this.label238.TabIndex = 180;
            this.label238.Text = "by";
            // 
            // switch2Flip
            // 
            this.switch2Flip.AutoSize = true;
            this.switch2Flip.Location = new System.Drawing.Point(277, 85);
            this.switch2Flip.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch2Flip.Name = "switch2Flip";
            this.switch2Flip.Size = new System.Drawing.Size(18, 17);
            this.switch2Flip.TabIndex = 152;
            this.switch2Flip.UseVisualStyleBackColor = true;
            this.switch2Flip.CheckedChanged += new System.EventHandler(this.switch2Flip_CheckedChanged);
            // 
            // switch5MappingBox
            // 
            this.switch5MappingBox.DisplayMember = "1";
            this.switch5MappingBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch5MappingBox.FormattingEnabled = true;
            this.switch5MappingBox.Items.AddRange(new object[] {
            "First to Smin",
            "Joint Position2",
            "Joint Position1",
            "Toggle"});
            this.switch5MappingBox.Location = new System.Drawing.Point(152, 158);
            this.switch5MappingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch5MappingBox.Name = "switch5MappingBox";
            this.switch5MappingBox.Size = new System.Drawing.Size(111, 24);
            this.switch5MappingBox.TabIndex = 183;
            this.switch5MappingBox.SelectedIndexChanged += new System.EventHandler(this.switch5MappingBox_SelectedIndexChanged);
            // 
            // switch4MappingBox
            // 
            this.switch4MappingBox.DisplayMember = "1";
            this.switch4MappingBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch4MappingBox.FormattingEnabled = true;
            this.switch4MappingBox.Items.AddRange(new object[] {
            "First to Smin",
            "Joint Position2",
            "Joint Position1",
            "Toggle"});
            this.switch4MappingBox.Location = new System.Drawing.Point(152, 133);
            this.switch4MappingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch4MappingBox.Name = "switch4MappingBox";
            this.switch4MappingBox.Size = new System.Drawing.Size(111, 24);
            this.switch4MappingBox.TabIndex = 181;
            this.switch4MappingBox.SelectedIndexChanged += new System.EventHandler(this.switch4MappingBox_SelectedIndexChanged);
            // 
            // switch2OutputBox
            // 
            this.switch2OutputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch2OutputBox.FormattingEnabled = true;
            this.switch2OutputBox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.switch2OutputBox.Location = new System.Drawing.Point(9, 81);
            this.switch2OutputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch2OutputBox.Name = "switch2OutputBox";
            this.switch2OutputBox.Size = new System.Drawing.Size(111, 24);
            this.switch2OutputBox.TabIndex = 70;
            this.switch2OutputBox.SelectedIndexChanged += new System.EventHandler(this.switch2OutputBox_SelectedIndexChanged);
            // 
            // switch3Flip
            // 
            this.switch3Flip.AutoSize = true;
            this.switch3Flip.Location = new System.Drawing.Point(277, 111);
            this.switch3Flip.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch3Flip.Name = "switch3Flip";
            this.switch3Flip.Size = new System.Drawing.Size(18, 17);
            this.switch3Flip.TabIndex = 153;
            this.switch3Flip.UseVisualStyleBackColor = true;
            this.switch3Flip.CheckedChanged += new System.EventHandler(this.switch3Flip_CheckedChanged);
            // 
            // label240
            // 
            this.label240.AutoSize = true;
            this.label240.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label240.Location = new System.Drawing.Point(127, 86);
            this.label240.Name = "label240";
            this.label240.Size = new System.Drawing.Size(23, 17);
            this.label240.TabIndex = 176;
            this.label240.Text = "by";
            // 
            // label253
            // 
            this.label253.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label253.Location = new System.Drawing.Point(272, 26);
            this.label253.Name = "label253";
            this.label253.Size = new System.Drawing.Size(35, 17);
            this.label253.TabIndex = 150;
            this.label253.Text = "Flip:";
            // 
            // label239
            // 
            this.label239.AutoSize = true;
            this.label239.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label239.Location = new System.Drawing.Point(127, 112);
            this.label239.Name = "label239";
            this.label239.Size = new System.Drawing.Size(23, 17);
            this.label239.TabIndex = 178;
            this.label239.Text = "by";
            // 
            // label38
            // 
            this.label38.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label38.Location = new System.Drawing.Point(148, 26);
            this.label38.Name = "label38";
            this.label38.Size = new System.Drawing.Size(108, 17);
            this.label38.TabIndex = 186;
            this.label38.Text = "Mapping:";
            // 
            // switch5Flip
            // 
            this.switch5Flip.AutoSize = true;
            this.switch5Flip.Location = new System.Drawing.Point(277, 161);
            this.switch5Flip.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch5Flip.Name = "switch5Flip";
            this.switch5Flip.Size = new System.Drawing.Size(18, 17);
            this.switch5Flip.TabIndex = 155;
            this.switch5Flip.UseVisualStyleBackColor = true;
            this.switch5Flip.CheckedChanged += new System.EventHandler(this.switch5Flip_CheckedChanged);
            // 
            // label241
            // 
            this.label241.AutoSize = true;
            this.label241.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label241.Location = new System.Drawing.Point(127, 59);
            this.label241.Name = "label241";
            this.label241.Size = new System.Drawing.Size(23, 17);
            this.label241.TabIndex = 135;
            this.label241.Text = "by";
            // 
            // label237
            // 
            this.label237.AutoSize = true;
            this.label237.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label237.Location = new System.Drawing.Point(127, 162);
            this.label237.Name = "label237";
            this.label237.Size = new System.Drawing.Size(23, 17);
            this.label237.TabIndex = 182;
            this.label237.Text = "by";
            // 
            // switch1MappingBox
            // 
            this.switch1MappingBox.DisplayMember = "1";
            this.switch1MappingBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch1MappingBox.FormattingEnabled = true;
            this.switch1MappingBox.Items.AddRange(new object[] {
            "First to Smin",
            "Joint Position2",
            "Joint Position1",
            "Toggle"});
            this.switch1MappingBox.Location = new System.Drawing.Point(152, 55);
            this.switch1MappingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch1MappingBox.Name = "switch1MappingBox";
            this.switch1MappingBox.Size = new System.Drawing.Size(111, 24);
            this.switch1MappingBox.TabIndex = 136;
            this.switch1MappingBox.SelectedIndexChanged += new System.EventHandler(this.switch1MappingBox_SelectedIndexChanged);
            // 
            // switch3OutputBox
            // 
            this.switch3OutputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch3OutputBox.FormattingEnabled = true;
            this.switch3OutputBox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.switch3OutputBox.Location = new System.Drawing.Point(9, 107);
            this.switch3OutputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch3OutputBox.Name = "switch3OutputBox";
            this.switch3OutputBox.Size = new System.Drawing.Size(111, 24);
            this.switch3OutputBox.TabIndex = 71;
            this.switch3OutputBox.SelectedIndexChanged += new System.EventHandler(this.switch3OutputBox_SelectedIndexChanged);
            // 
            // switch4Flip
            // 
            this.switch4Flip.AutoSize = true;
            this.switch4Flip.Location = new System.Drawing.Point(277, 137);
            this.switch4Flip.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch4Flip.Name = "switch4Flip";
            this.switch4Flip.Size = new System.Drawing.Size(18, 17);
            this.switch4Flip.TabIndex = 154;
            this.switch4Flip.UseVisualStyleBackColor = true;
            this.switch4Flip.CheckedChanged += new System.EventHandler(this.switch4Flip_CheckedChanged);
            // 
            // switch3MappingBox
            // 
            this.switch3MappingBox.DisplayMember = "1";
            this.switch3MappingBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch3MappingBox.FormattingEnabled = true;
            this.switch3MappingBox.Items.AddRange(new object[] {
            "First to Smin",
            "Joint Position2",
            "Joint Position1",
            "Toggle"});
            this.switch3MappingBox.Location = new System.Drawing.Point(152, 107);
            this.switch3MappingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch3MappingBox.Name = "switch3MappingBox";
            this.switch3MappingBox.Size = new System.Drawing.Size(111, 24);
            this.switch3MappingBox.TabIndex = 179;
            this.switch3MappingBox.SelectedIndexChanged += new System.EventHandler(this.switch3MappingBox_SelectedIndexChanged);
            // 
            // switch5OutputBox
            // 
            this.switch5OutputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch5OutputBox.FormattingEnabled = true;
            this.switch5OutputBox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.switch5OutputBox.Location = new System.Drawing.Point(9, 158);
            this.switch5OutputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch5OutputBox.Name = "switch5OutputBox";
            this.switch5OutputBox.Size = new System.Drawing.Size(111, 24);
            this.switch5OutputBox.TabIndex = 73;
            this.switch5OutputBox.SelectedIndexChanged += new System.EventHandler(this.switch5OutputBox_SelectedIndexChanged);
            // 
            // switch2MappingBox
            // 
            this.switch2MappingBox.DisplayMember = "1";
            this.switch2MappingBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch2MappingBox.FormattingEnabled = true;
            this.switch2MappingBox.Items.AddRange(new object[] {
            "First to Smin",
            "Joint Position2",
            "Joint Position1",
            "Toggle"});
            this.switch2MappingBox.Location = new System.Drawing.Point(152, 81);
            this.switch2MappingBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch2MappingBox.Name = "switch2MappingBox";
            this.switch2MappingBox.Size = new System.Drawing.Size(111, 24);
            this.switch2MappingBox.TabIndex = 177;
            this.switch2MappingBox.SelectedIndexChanged += new System.EventHandler(this.switch2MappingBox_SelectedIndexChanged);
            // 
            // switch4OutputBox
            // 
            this.switch4OutputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switch4OutputBox.FormattingEnabled = true;
            this.switch4OutputBox.Items.AddRange(new object[] {
            "Off",
            "Shoulder",
            "Elbow",
            "Wrist Rotate",
            "Wrist Flex",
            "Hand"});
            this.switch4OutputBox.Location = new System.Drawing.Point(9, 133);
            this.switch4OutputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switch4OutputBox.Name = "switch4OutputBox";
            this.switch4OutputBox.Size = new System.Drawing.Size(111, 24);
            this.switch4OutputBox.TabIndex = 72;
            this.switch4OutputBox.SelectedIndexChanged += new System.EventHandler(this.switch4OutputBox_SelectedIndexChanged);
            // 
            // label148
            // 
            this.label148.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label148.Location = new System.Drawing.Point(361, 539);
            this.label148.Name = "label148";
            this.label148.Size = new System.Drawing.Size(107, 18);
            this.label148.TabIndex = 223;
            this.label148.Text = "ID2 State:";
            this.label148.TextAlign = System.Drawing.ContentAlignment.TopRight;
            this.label148.Visible = false;
            // 
            // label103
            // 
            this.label103.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label103.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label103.Location = new System.Drawing.Point(213, 76);
            this.label103.Name = "label103";
            this.label103.Size = new System.Drawing.Size(56, 18);
            this.label103.TabIndex = 198;
            this.label103.Text = "Gain:";
            // 
            // label104
            // 
            this.label104.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label104.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label104.Location = new System.Drawing.Point(16, 76);
            this.label104.Name = "label104";
            this.label104.Size = new System.Drawing.Size(124, 18);
            this.label104.TabIndex = 199;
            this.label104.Text = "Signal Strength:";
            // 
            // label145
            // 
            this.label145.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label145.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label145.Location = new System.Drawing.Point(276, 76);
            this.label145.Name = "label145";
            this.label145.Size = new System.Drawing.Size(47, 18);
            this.label145.TabIndex = 200;
            this.label145.Text = "Smin:";
            // 
            // label147
            // 
            this.label147.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label147.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label147.Location = new System.Drawing.Point(333, 76);
            this.label147.Name = "label147";
            this.label147.Size = new System.Drawing.Size(49, 18);
            this.label147.TabIndex = 201;
            this.label147.Text = "Smax:";
            // 
            // switchSmaxLabel1
            // 
            this.switchSmaxLabel1.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.switchSmaxLabel1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSmaxLabel1.Location = new System.Drawing.Point(179, 128);
            this.switchSmaxLabel1.Name = "switchSmaxLabel1";
            this.switchSmaxLabel1.Size = new System.Drawing.Size(40, 18);
            this.switchSmaxLabel1.TabIndex = 197;
            this.switchSmaxLabel1.Text = "Smax";
            this.switchSmaxLabel1.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // switchSminLabel1
            // 
            this.switchSminLabel1.Font = new System.Drawing.Font("Microsoft Sans Serif", 6.75F);
            this.switchSminLabel1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSminLabel1.Location = new System.Drawing.Point(4, 128);
            this.switchSminLabel1.Name = "switchSminLabel1";
            this.switchSminLabel1.Size = new System.Drawing.Size(35, 18);
            this.switchSminLabel1.TabIndex = 196;
            this.switchSminLabel1.Text = "Smin";
            this.switchSminLabel1.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // switchSminTick1
            // 
            this.switchSminTick1.BackColor = System.Drawing.Color.MediumPurple;
            this.switchSminTick1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSminTick1.Location = new System.Drawing.Point(20, 98);
            this.switchSminTick1.Name = "switchSminTick1";
            this.switchSminTick1.Size = new System.Drawing.Size(3, 30);
            this.switchSminTick1.TabIndex = 195;
            // 
            // switchSmaxTick1
            // 
            this.switchSmaxTick1.BackColor = System.Drawing.Color.MediumPurple;
            this.switchSmaxTick1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSmaxTick1.Location = new System.Drawing.Point(197, 98);
            this.switchSmaxTick1.Name = "switchSmaxTick1";
            this.switchSmaxTick1.Size = new System.Drawing.Size(3, 30);
            this.switchSmaxTick1.TabIndex = 194;
            // 
            // switchSmaxCtrl1
            // 
            this.switchSmaxCtrl1.DecimalPlaces = 1;
            this.switchSmaxCtrl1.Increment = new decimal(new int[] {
            2,
            0,
            0,
            65536});
            this.switchSmaxCtrl1.Location = new System.Drawing.Point(336, 98);
            this.switchSmaxCtrl1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSmaxCtrl1.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSmaxCtrl1.Name = "switchSmaxCtrl1";
            this.switchSmaxCtrl1.Size = new System.Drawing.Size(53, 22);
            this.switchSmaxCtrl1.TabIndex = 193;
            this.switchSmaxCtrl1.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSmaxCtrl1.ValueChanged += new System.EventHandler(this.switchSmaxCtrl1_ValueChanged);
            // 
            // switchSminCtrl1
            // 
            this.switchSminCtrl1.DecimalPlaces = 1;
            this.switchSminCtrl1.Increment = new decimal(new int[] {
            2,
            0,
            0,
            65536});
            this.switchSminCtrl1.Location = new System.Drawing.Point(279, 98);
            this.switchSminCtrl1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSminCtrl1.Maximum = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.switchSminCtrl1.Name = "switchSminCtrl1";
            this.switchSminCtrl1.Size = new System.Drawing.Size(53, 22);
            this.switchSminCtrl1.TabIndex = 192;
            this.switchSminCtrl1.ValueChanged += new System.EventHandler(this.switchSminCtrl1_ValueChanged);
            // 
            // switchSignalBar1
            // 
            this.switchSignalBar1.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchSignalBar1.Location = new System.Drawing.Point(20, 98);
            this.switchSignalBar1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchSignalBar1.MarqueeAnimationSpeed = 30;
            this.switchSignalBar1.Maximum = 500;
            this.switchSignalBar1.Name = "switchSignalBar1";
            this.switchSignalBar1.Size = new System.Drawing.Size(179, 27);
            this.switchSignalBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.switchSignalBar1.TabIndex = 191;
            // 
            // switchGainCtrl1
            // 
            this.switchGainCtrl1.Location = new System.Drawing.Point(216, 98);
            this.switchGainCtrl1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchGainCtrl1.Maximum = new decimal(new int[] {
            5000,
            0,
            0,
            0});
            this.switchGainCtrl1.Name = "switchGainCtrl1";
            this.switchGainCtrl1.Size = new System.Drawing.Size(56, 22);
            this.switchGainCtrl1.TabIndex = 190;
            this.switchGainCtrl1.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.switchGainCtrl1.ValueChanged += new System.EventHandler(this.switchGainCtrl1_ValueChanged);
            // 
            // switchInputBox
            // 
            this.switchInputBox.DisplayMember = "1";
            this.switchInputBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switchInputBox.FormattingEnabled = true;
            this.switchInputBox.Items.AddRange(new object[] {
            "Momentary Button",
            "Co-contraction"});
            this.switchInputBox.Location = new System.Drawing.Point(325, 42);
            this.switchInputBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchInputBox.Name = "switchInputBox";
            this.switchInputBox.Size = new System.Drawing.Size(129, 24);
            this.switchInputBox.TabIndex = 188;
            this.switchInputBox.SelectedIndexChanged += new System.EventHandler(this.switchInputBox_SelectedIndexChanged);
            // 
            // label39
            // 
            this.label39.AutoSize = true;
            this.label39.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label39.Location = new System.Drawing.Point(321, 23);
            this.label39.Name = "label39";
            this.label39.Size = new System.Drawing.Size(90, 17);
            this.label39.TabIndex = 187;
            this.label39.Text = "Input Device:";
            // 
            // label27
            // 
            this.label27.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label27.Location = new System.Drawing.Point(16, 23);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(135, 18);
            this.label27.TabIndex = 184;
            this.label27.Text = "Degree of Freedom:";
            // 
            // switchTimeCtrl1
            // 
            this.switchTimeCtrl1.Location = new System.Drawing.Point(397, 98);
            this.switchTimeCtrl1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchTimeCtrl1.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.switchTimeCtrl1.Name = "switchTimeCtrl1";
            this.switchTimeCtrl1.Size = new System.Drawing.Size(61, 22);
            this.switchTimeCtrl1.TabIndex = 175;
            this.switchTimeCtrl1.Value = new decimal(new int[] {
            40,
            0,
            0,
            0});
            this.switchTimeCtrl1.ValueChanged += new System.EventHandler(this.switchTimeCtrl1_ValueChanged);
            // 
            // label242
            // 
            this.label242.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label242.Location = new System.Drawing.Point(395, 76);
            this.label242.Name = "label242";
            this.label242.Size = new System.Drawing.Size(61, 18);
            this.label242.TabIndex = 174;
            this.label242.Text = "CCtime:";
            // 
            // switchModeBox
            // 
            this.switchModeBox.DisplayMember = "1";
            this.switchModeBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switchModeBox.FormattingEnabled = true;
            this.switchModeBox.Items.AddRange(new object[] {
            "Button Press",
            "Co-contraction"});
            this.switchModeBox.Location = new System.Drawing.Point(172, 43);
            this.switchModeBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchModeBox.Name = "switchModeBox";
            this.switchModeBox.Size = new System.Drawing.Size(129, 24);
            this.switchModeBox.TabIndex = 148;
            this.switchModeBox.SelectedIndexChanged += new System.EventHandler(this.switchModeBox_SelectedIndexChanged);
            // 
            // switchLabel
            // 
            this.switchLabel.AutoSize = true;
            this.switchLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 28.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.switchLabel.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.switchLabel.Location = new System.Drawing.Point(105, 405);
            this.switchLabel.Name = "switchLabel";
            this.switchLabel.Size = new System.Drawing.Size(56, 55);
            this.switchLabel.TabIndex = 69;
            this.switchLabel.Text = "--";
            // 
            // label257
            // 
            this.label257.AutoSize = true;
            this.label257.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label257.Location = new System.Drawing.Point(12, 426);
            this.label257.Name = "label257";
            this.label257.Size = new System.Drawing.Size(84, 17);
            this.label257.TabIndex = 68;
            this.label257.Text = "Switched to:";
            // 
            // switchDoFbox
            // 
            this.switchDoFbox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.switchDoFbox.FormattingEnabled = true;
            this.switchDoFbox.Items.AddRange(new object[] {
            "Off",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6"});
            this.switchDoFbox.Location = new System.Drawing.Point(20, 43);
            this.switchDoFbox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.switchDoFbox.Name = "switchDoFbox";
            this.switchDoFbox.Size = new System.Drawing.Size(129, 24);
            this.switchDoFbox.TabIndex = 19;
            this.switchDoFbox.SelectedIndexChanged += new System.EventHandler(this.switchDoFbox_SelectedIndexChanged);
            this.switchDoFbox.Enter += new System.EventHandler(this.switchDoFbox_Enter);
            // 
            // label258
            // 
            this.label258.AutoSize = true;
            this.label258.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label258.Location = new System.Drawing.Point(169, 25);
            this.label258.Name = "label258";
            this.label258.Size = new System.Drawing.Size(91, 17);
            this.label258.TabIndex = 18;
            this.label258.Text = "Switch Mode:";
            // 
            // label162
            // 
            this.label162.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label162.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label162.Location = new System.Drawing.Point(333, 5);
            this.label162.Name = "label162";
            this.label162.Size = new System.Drawing.Size(136, 18);
            this.label162.TabIndex = 134;
            this.label162.Text = "Mapping";
            // 
            // label150
            // 
            this.label150.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label150.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label150.Location = new System.Drawing.Point(173, 5);
            this.label150.Name = "label150";
            this.label150.Size = new System.Drawing.Size(136, 18);
            this.label150.TabIndex = 133;
            this.label150.Text = "Output Device";
            // 
            // label146
            // 
            this.label146.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label146.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label146.Location = new System.Drawing.Point(13, 5);
            this.label146.Name = "label146";
            this.label146.Size = new System.Drawing.Size(132, 18);
            this.label146.TabIndex = 132;
            this.label146.Text = "Input Device";
            // 
            // label163
            // 
            this.label163.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label163.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label163.Location = new System.Drawing.Point(692, 5);
            this.label163.Name = "label163";
            this.label163.Size = new System.Drawing.Size(56, 18);
            this.label163.TabIndex = 14;
            this.label163.Text = "Gain";
            // 
            // label158
            // 
            this.label158.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label158.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label158.Location = new System.Drawing.Point(493, 5);
            this.label158.Name = "label158";
            this.label158.Size = new System.Drawing.Size(124, 18);
            this.label158.TabIndex = 25;
            this.label158.Text = "Signal Strength";
            // 
            // label157
            // 
            this.label157.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label157.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label157.Location = new System.Drawing.Point(753, 5);
            this.label157.Name = "label157";
            this.label157.Size = new System.Drawing.Size(47, 18);
            this.label157.TabIndex = 26;
            this.label157.Text = "Smin";
            // 
            // label156
            // 
            this.label156.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Underline, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label156.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label156.Location = new System.Drawing.Point(812, 5);
            this.label156.Name = "label156";
            this.label156.Size = new System.Drawing.Size(49, 18);
            this.label156.TabIndex = 29;
            this.label156.Text = "Smax";
            // 
            // doF6
            // 
            this.doF6.Location = new System.Drawing.Point(8, 630);
            this.doF6.Margin = new System.Windows.Forms.Padding(5);
            this.doF6.Name = "doF6";
            this.doF6.Size = new System.Drawing.Size(865, 116);
            this.doF6.TabIndex = 144;
            // 
            // doF5
            // 
            this.doF5.Location = new System.Drawing.Point(8, 507);
            this.doF5.Margin = new System.Windows.Forms.Padding(5);
            this.doF5.Name = "doF5";
            this.doF5.Size = new System.Drawing.Size(865, 116);
            this.doF5.TabIndex = 143;
            // 
            // doF4
            // 
            this.doF4.Location = new System.Drawing.Point(8, 384);
            this.doF4.Margin = new System.Windows.Forms.Padding(5);
            this.doF4.Name = "doF4";
            this.doF4.Size = new System.Drawing.Size(865, 116);
            this.doF4.TabIndex = 142;
            // 
            // doF3
            // 
            this.doF3.Location = new System.Drawing.Point(8, 261);
            this.doF3.Margin = new System.Windows.Forms.Padding(5);
            this.doF3.Name = "doF3";
            this.doF3.Size = new System.Drawing.Size(865, 116);
            this.doF3.TabIndex = 141;
            // 
            // doF2
            // 
            this.doF2.Location = new System.Drawing.Point(8, 142);
            this.doF2.Margin = new System.Windows.Forms.Padding(5);
            this.doF2.Name = "doF2";
            this.doF2.Size = new System.Drawing.Size(865, 116);
            this.doF2.TabIndex = 140;
            // 
            // doF1
            // 
            this.doF1.Location = new System.Drawing.Point(8, 22);
            this.doF1.Margin = new System.Windows.Forms.Padding(5);
            this.doF1.Name = "doF1";
            this.doF1.Size = new System.Drawing.Size(865, 116);
            this.doF1.TabIndex = 139;
            // 
            // tabBento
            // 
            this.tabBento.Controls.Add(this.AutoLevellingBox);
            this.tabBento.Controls.Add(this.groupBox19);
            this.tabBento.Controls.Add(this.BentoEnvLimitsBox);
            this.tabBento.Controls.Add(this.BentoAdaptGripBox);
            this.tabBento.Controls.Add(this.SimBox);
            this.tabBento.Controls.Add(this.BentoGroupBox);
            this.tabBento.Controls.Add(this.RobotParamBox);
            this.tabBento.Controls.Add(this.RobotFeedbackBox);
            this.tabBento.Controls.Add(this.LEDon);
            this.tabBento.Controls.Add(this.LEDoff);
            this.tabBento.Controls.Add(this.dynaError);
            this.tabBento.Controls.Add(this.moveCW);
            this.tabBento.Controls.Add(this.label120);
            this.tabBento.Controls.Add(this.moveCCW);
            this.tabBento.Controls.Add(this.dynaCommResult);
            this.tabBento.Controls.Add(this.readFeedback);
            this.tabBento.Controls.Add(this.label118);
            this.tabBento.Location = new System.Drawing.Point(4, 25);
            this.tabBento.Margin = new System.Windows.Forms.Padding(4);
            this.tabBento.Name = "tabBento";
            this.tabBento.Size = new System.Drawing.Size(1552, 793);
            this.tabBento.TabIndex = 2;
            this.tabBento.Text = "Bento Arm";
            this.tabBento.UseVisualStyleBackColor = true;
            // 
            // AutoLevellingBox
            // 
            this.AutoLevellingBox.Controls.Add(this.LogPID_Enabled);
            this.AutoLevellingBox.Controls.Add(this.NN_PID_Enabled);
            this.AutoLevellingBox.Controls.Add(this.FlexionPIDBox);
            this.AutoLevellingBox.Controls.Add(this.RotationPIDBox);
            this.AutoLevellingBox.Controls.Add(this.AL_Enabled);
            this.AutoLevellingBox.Enabled = false;
            this.AutoLevellingBox.Location = new System.Drawing.Point(4, 354);
            this.AutoLevellingBox.Margin = new System.Windows.Forms.Padding(4);
            this.AutoLevellingBox.Name = "AutoLevellingBox";
            this.AutoLevellingBox.Padding = new System.Windows.Forms.Padding(4);
            this.AutoLevellingBox.Size = new System.Drawing.Size(489, 181);
            this.AutoLevellingBox.TabIndex = 211;
            this.AutoLevellingBox.TabStop = false;
            this.AutoLevellingBox.Text = "Auto-Levelling";
            // 
            // NN_PID_Enabled
            // 
            this.NN_PID_Enabled.AutoSize = true;
            this.NN_PID_Enabled.Location = new System.Drawing.Point(11, 54);
            this.NN_PID_Enabled.Margin = new System.Windows.Forms.Padding(4);
            this.NN_PID_Enabled.Name = "NN_PID_Enabled";
            this.NN_PID_Enabled.Size = new System.Drawing.Size(76, 21);
            this.NN_PID_Enabled.TabIndex = 216;
            this.NN_PID_Enabled.Text = "NN PID";
            this.NN_PID_Enabled.UseVisualStyleBackColor = true;
            // 
            // FlexionPIDBox
            // 
            this.FlexionPIDBox.Controls.Add(this.CurrentFlexion);
            this.FlexionPIDBox.Controls.Add(this.label224);
            this.FlexionPIDBox.Controls.Add(this.label223);
            this.FlexionPIDBox.Controls.Add(this.SetpointFlexion);
            this.FlexionPIDBox.Controls.Add(this.Kd_theta_ctrl);
            this.FlexionPIDBox.Controls.Add(this.label210);
            this.FlexionPIDBox.Controls.Add(this.Ki_theta_ctrl);
            this.FlexionPIDBox.Controls.Add(this.label206);
            this.FlexionPIDBox.Controls.Add(this.Kp_theta_ctrl);
            this.FlexionPIDBox.Controls.Add(this.label212);
            this.FlexionPIDBox.Enabled = false;
            this.FlexionPIDBox.Location = new System.Drawing.Point(299, 13);
            this.FlexionPIDBox.Margin = new System.Windows.Forms.Padding(4);
            this.FlexionPIDBox.Name = "FlexionPIDBox";
            this.FlexionPIDBox.Padding = new System.Windows.Forms.Padding(4);
            this.FlexionPIDBox.Size = new System.Drawing.Size(182, 156);
            this.FlexionPIDBox.TabIndex = 212;
            this.FlexionPIDBox.TabStop = false;
            this.FlexionPIDBox.Text = "Flexion PID";
            // 
            // CurrentFlexion
            // 
            this.CurrentFlexion.AutoSize = true;
            this.CurrentFlexion.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.CurrentFlexion.Location = new System.Drawing.Point(65, 134);
            this.CurrentFlexion.Name = "CurrentFlexion";
            this.CurrentFlexion.Size = new System.Drawing.Size(18, 17);
            this.CurrentFlexion.TabIndex = 217;
            this.CurrentFlexion.Text = "--";
            // 
            // label224
            // 
            this.label224.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label224.Location = new System.Drawing.Point(6, 132);
            this.label224.Name = "label224";
            this.label224.Size = new System.Drawing.Size(58, 17);
            this.label224.TabIndex = 216;
            this.label224.Text = "Current";
            // 
            // label223
            // 
            this.label223.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label223.Location = new System.Drawing.Point(5, 107);
            this.label223.Name = "label223";
            this.label223.Size = new System.Drawing.Size(60, 22);
            this.label223.TabIndex = 215;
            this.label223.Text = "Setpoint";
            // 
            // SetpointFlexion
            // 
            this.SetpointFlexion.AutoSize = true;
            this.SetpointFlexion.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.SetpointFlexion.Location = new System.Drawing.Point(65, 107);
            this.SetpointFlexion.Name = "SetpointFlexion";
            this.SetpointFlexion.Size = new System.Drawing.Size(18, 17);
            this.SetpointFlexion.TabIndex = 214;
            this.SetpointFlexion.Text = "--";
            // 
            // Kd_theta_ctrl
            // 
            this.Kd_theta_ctrl.DecimalPlaces = 5;
            this.Kd_theta_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.Kd_theta_ctrl.Location = new System.Drawing.Point(42, 73);
            this.Kd_theta_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Kd_theta_ctrl.Name = "Kd_theta_ctrl";
            this.Kd_theta_ctrl.Size = new System.Drawing.Size(97, 22);
            this.Kd_theta_ctrl.TabIndex = 153;
            this.Kd_theta_ctrl.Value = new decimal(new int[] {
            8,
            0,
            0,
            196608});
            this.Kd_theta_ctrl.ValueChanged += new System.EventHandler(this.Kd_theta_ctrl_ValueChanged);
            // 
            // label210
            // 
            this.label210.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label210.Location = new System.Drawing.Point(5, 73);
            this.label210.Name = "label210";
            this.label210.Size = new System.Drawing.Size(31, 22);
            this.label210.TabIndex = 154;
            this.label210.Text = "Kd";
            // 
            // Ki_theta_ctrl
            // 
            this.Ki_theta_ctrl.DecimalPlaces = 5;
            this.Ki_theta_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.Ki_theta_ctrl.Location = new System.Drawing.Point(42, 47);
            this.Ki_theta_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Ki_theta_ctrl.Name = "Ki_theta_ctrl";
            this.Ki_theta_ctrl.Size = new System.Drawing.Size(97, 22);
            this.Ki_theta_ctrl.TabIndex = 151;
            this.Ki_theta_ctrl.Value = new decimal(new int[] {
            42,
            0,
            0,
            131072});
            this.Ki_theta_ctrl.ValueChanged += new System.EventHandler(this.Ki_theta_ctrl_ValueChanged);
            // 
            // label206
            // 
            this.label206.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label206.Location = new System.Drawing.Point(5, 47);
            this.label206.Name = "label206";
            this.label206.Size = new System.Drawing.Size(31, 22);
            this.label206.TabIndex = 152;
            this.label206.Text = "Ki";
            // 
            // Kp_theta_ctrl
            // 
            this.Kp_theta_ctrl.DecimalPlaces = 5;
            this.Kp_theta_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.Kp_theta_ctrl.Location = new System.Drawing.Point(42, 21);
            this.Kp_theta_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Kp_theta_ctrl.Name = "Kp_theta_ctrl";
            this.Kp_theta_ctrl.Size = new System.Drawing.Size(97, 22);
            this.Kp_theta_ctrl.TabIndex = 150;
            this.Kp_theta_ctrl.Value = new decimal(new int[] {
            29,
            0,
            0,
            131072});
            this.Kp_theta_ctrl.ValueChanged += new System.EventHandler(this.Kp_theta_ctrl_ValueChanged);
            // 
            // label212
            // 
            this.label212.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label212.Location = new System.Drawing.Point(5, 21);
            this.label212.Name = "label212";
            this.label212.Size = new System.Drawing.Size(31, 22);
            this.label212.TabIndex = 150;
            this.label212.Text = "Kp";
            // 
            // RotationPIDBox
            // 
            this.RotationPIDBox.Controls.Add(this.CurrentRotation);
            this.RotationPIDBox.Controls.Add(this.label222);
            this.RotationPIDBox.Controls.Add(this.label221);
            this.RotationPIDBox.Controls.Add(this.Kd_phi_ctrl);
            this.RotationPIDBox.Controls.Add(this.SetpointRotation);
            this.RotationPIDBox.Controls.Add(this.label215);
            this.RotationPIDBox.Controls.Add(this.Ki_phi_ctrl);
            this.RotationPIDBox.Controls.Add(this.label225);
            this.RotationPIDBox.Controls.Add(this.Kp_phi_ctrl);
            this.RotationPIDBox.Controls.Add(this.label226);
            this.RotationPIDBox.Enabled = false;
            this.RotationPIDBox.Location = new System.Drawing.Point(108, 13);
            this.RotationPIDBox.Margin = new System.Windows.Forms.Padding(4);
            this.RotationPIDBox.Name = "RotationPIDBox";
            this.RotationPIDBox.Padding = new System.Windows.Forms.Padding(4);
            this.RotationPIDBox.Size = new System.Drawing.Size(183, 156);
            this.RotationPIDBox.TabIndex = 211;
            this.RotationPIDBox.TabStop = false;
            this.RotationPIDBox.Text = "Rotation PID";
            // 
            // CurrentRotation
            // 
            this.CurrentRotation.AutoSize = true;
            this.CurrentRotation.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.CurrentRotation.Location = new System.Drawing.Point(65, 134);
            this.CurrentRotation.Name = "CurrentRotation";
            this.CurrentRotation.Size = new System.Drawing.Size(18, 17);
            this.CurrentRotation.TabIndex = 216;
            this.CurrentRotation.Text = "--";
            // 
            // label222
            // 
            this.label222.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label222.Location = new System.Drawing.Point(7, 132);
            this.label222.Name = "label222";
            this.label222.Size = new System.Drawing.Size(63, 19);
            this.label222.TabIndex = 215;
            this.label222.Text = "Current";
            // 
            // label221
            // 
            this.label221.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label221.Location = new System.Drawing.Point(7, 107);
            this.label221.Name = "label221";
            this.label221.Size = new System.Drawing.Size(63, 22);
            this.label221.TabIndex = 214;
            this.label221.Text = "Setpoint";
            // 
            // Kd_phi_ctrl
            // 
            this.Kd_phi_ctrl.DecimalPlaces = 5;
            this.Kd_phi_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.Kd_phi_ctrl.Location = new System.Drawing.Point(42, 73);
            this.Kd_phi_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Kd_phi_ctrl.Name = "Kd_phi_ctrl";
            this.Kd_phi_ctrl.Size = new System.Drawing.Size(102, 22);
            this.Kd_phi_ctrl.TabIndex = 153;
            this.Kd_phi_ctrl.Value = new decimal(new int[] {
            879,
            0,
            0,
            327680});
            this.Kd_phi_ctrl.ValueChanged += new System.EventHandler(this.Kd_phi_ctrl_ValueChanged);
            // 
            // SetpointRotation
            // 
            this.SetpointRotation.AutoSize = true;
            this.SetpointRotation.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.SetpointRotation.Location = new System.Drawing.Point(65, 107);
            this.SetpointRotation.Name = "SetpointRotation";
            this.SetpointRotation.Size = new System.Drawing.Size(18, 17);
            this.SetpointRotation.TabIndex = 213;
            this.SetpointRotation.Text = "--";
            // 
            // label215
            // 
            this.label215.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label215.Location = new System.Drawing.Point(5, 73);
            this.label215.Name = "label215";
            this.label215.Size = new System.Drawing.Size(31, 22);
            this.label215.TabIndex = 154;
            this.label215.Text = "Kd";
            // 
            // Ki_phi_ctrl
            // 
            this.Ki_phi_ctrl.DecimalPlaces = 5;
            this.Ki_phi_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.Ki_phi_ctrl.Location = new System.Drawing.Point(42, 47);
            this.Ki_phi_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Ki_phi_ctrl.Name = "Ki_phi_ctrl";
            this.Ki_phi_ctrl.Size = new System.Drawing.Size(102, 22);
            this.Ki_phi_ctrl.TabIndex = 151;
            this.Ki_phi_ctrl.Value = new decimal(new int[] {
            6,
            0,
            0,
            131072});
            this.Ki_phi_ctrl.ValueChanged += new System.EventHandler(this.Ki_phi_ctrl_ValueChanged);
            // 
            // label225
            // 
            this.label225.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label225.Location = new System.Drawing.Point(5, 47);
            this.label225.Name = "label225";
            this.label225.Size = new System.Drawing.Size(31, 22);
            this.label225.TabIndex = 152;
            this.label225.Text = "Ki";
            // 
            // Kp_phi_ctrl
            // 
            this.Kp_phi_ctrl.DecimalPlaces = 5;
            this.Kp_phi_ctrl.Increment = new decimal(new int[] {
            1,
            0,
            0,
            131072});
            this.Kp_phi_ctrl.Location = new System.Drawing.Point(42, 21);
            this.Kp_phi_ctrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Kp_phi_ctrl.Name = "Kp_phi_ctrl";
            this.Kp_phi_ctrl.Size = new System.Drawing.Size(102, 22);
            this.Kp_phi_ctrl.TabIndex = 150;
            this.Kp_phi_ctrl.Value = new decimal(new int[] {
            32,
            0,
            0,
            131072});
            this.Kp_phi_ctrl.ValueChanged += new System.EventHandler(this.Kp_phi_ctrl_ValueChanged);
            // 
            // label226
            // 
            this.label226.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label226.Location = new System.Drawing.Point(5, 21);
            this.label226.Name = "label226";
            this.label226.Size = new System.Drawing.Size(31, 22);
            this.label226.TabIndex = 150;
            this.label226.Text = "Kp";
            // 
            // AL_Enabled
            // 
            this.AL_Enabled.AutoSize = true;
            this.AL_Enabled.Location = new System.Drawing.Point(11, 25);
            this.AL_Enabled.Margin = new System.Windows.Forms.Padding(4);
            this.AL_Enabled.Name = "AL_Enabled";
            this.AL_Enabled.Size = new System.Drawing.Size(82, 21);
            this.AL_Enabled.TabIndex = 202;
            this.AL_Enabled.Text = "Enabled";
            this.AL_Enabled.UseVisualStyleBackColor = true;
            // 
            // groupBox19
            // 
            this.groupBox19.Controls.Add(this.BentoProfileOpen);
            this.groupBox19.Controls.Add(this.BentoProfileBox);
            this.groupBox19.Controls.Add(this.BentoProfileSave);
            this.groupBox19.Location = new System.Drawing.Point(501, 354);
            this.groupBox19.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox19.Name = "groupBox19";
            this.groupBox19.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox19.Size = new System.Drawing.Size(303, 96);
            this.groupBox19.TabIndex = 210;
            this.groupBox19.TabStop = false;
            this.groupBox19.Text = "Joint Limit Profiles";
            // 
            // BentoProfileOpen
            // 
            this.BentoProfileOpen.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.BentoProfileOpen.Location = new System.Drawing.Point(224, 36);
            this.BentoProfileOpen.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoProfileOpen.Name = "BentoProfileOpen";
            this.BentoProfileOpen.Size = new System.Drawing.Size(59, 28);
            this.BentoProfileOpen.TabIndex = 13;
            this.BentoProfileOpen.Text = "Open";
            this.BentoProfileOpen.Click += new System.EventHandler(this.BentoProfileOpen_Click);
            // 
            // BentoProfileBox
            // 
            this.BentoProfileBox.DisplayMember = "1";
            this.BentoProfileBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.BentoProfileBox.FormattingEnabled = true;
            this.BentoProfileBox.Items.AddRange(new object[] {
            "profile0",
            "profile1",
            "profile2",
            "profile3",
            "profile4",
            "profile5",
            "profile6",
            "profile7",
            "profile8",
            "profile9"});
            this.BentoProfileBox.Location = new System.Drawing.Point(9, 38);
            this.BentoProfileBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoProfileBox.Name = "BentoProfileBox";
            this.BentoProfileBox.Size = new System.Drawing.Size(129, 24);
            this.BentoProfileBox.TabIndex = 189;
            // 
            // BentoProfileSave
            // 
            this.BentoProfileSave.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.BentoProfileSave.Location = new System.Drawing.Point(160, 36);
            this.BentoProfileSave.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoProfileSave.Name = "BentoProfileSave";
            this.BentoProfileSave.Size = new System.Drawing.Size(59, 28);
            this.BentoProfileSave.TabIndex = 14;
            this.BentoProfileSave.Text = "Save";
            this.BentoProfileSave.Click += new System.EventHandler(this.BentoProfileSave_Click);
            // 
            // BentoEnvLimitsBox
            // 
            this.BentoEnvLimitsBox.Controls.Add(this.label159);
            this.BentoEnvLimitsBox.Controls.Add(this.label155);
            this.BentoEnvLimitsBox.Controls.Add(this.numericUpDown3);
            this.BentoEnvLimitsBox.Controls.Add(this.label153);
            this.BentoEnvLimitsBox.Controls.Add(this.environCheck);
            this.BentoEnvLimitsBox.Controls.Add(this.numericUpDown2);
            this.BentoEnvLimitsBox.Controls.Add(this.label154);
            this.BentoEnvLimitsBox.Enabled = false;
            this.BentoEnvLimitsBox.Location = new System.Drawing.Point(4, 543);
            this.BentoEnvLimitsBox.Margin = new System.Windows.Forms.Padding(4);
            this.BentoEnvLimitsBox.Name = "BentoEnvLimitsBox";
            this.BentoEnvLimitsBox.Padding = new System.Windows.Forms.Padding(4);
            this.BentoEnvLimitsBox.Size = new System.Drawing.Size(492, 187);
            this.BentoEnvLimitsBox.TabIndex = 209;
            this.BentoEnvLimitsBox.TabStop = false;
            this.BentoEnvLimitsBox.Text = "Environment Limits";
            this.BentoEnvLimitsBox.Visible = false;
            // 
            // label159
            // 
            this.label159.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label159.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label159.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label159.Location = new System.Drawing.Point(7, 160);
            this.label159.Name = "label159";
            this.label159.Size = new System.Drawing.Size(475, 23);
            this.label159.TabIndex = 204;
            this.label159.Text = "Enable environment limits to prevent the arm from hitting the floor/stand.";
            // 
            // label155
            // 
            this.label155.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label155.Location = new System.Drawing.Point(107, 55);
            this.label155.Name = "label155";
            this.label155.Size = new System.Drawing.Size(83, 18);
            this.label155.TabIndex = 205;
            this.label155.Text = "Value (mm):";
            // 
            // numericUpDown3
            // 
            this.numericUpDown3.Location = new System.Drawing.Point(112, 106);
            this.numericUpDown3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDown3.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDown3.Name = "numericUpDown3";
            this.numericUpDown3.Size = new System.Drawing.Size(64, 22);
            this.numericUpDown3.TabIndex = 203;
            // 
            // label153
            // 
            this.label153.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label153.Location = new System.Drawing.Point(7, 108);
            this.label153.Name = "label153";
            this.label153.Size = new System.Drawing.Size(115, 18);
            this.label153.TabIndex = 204;
            this.label153.Text = "Backplane (B):";
            // 
            // environCheck
            // 
            this.environCheck.AutoSize = true;
            this.environCheck.Location = new System.Drawing.Point(11, 25);
            this.environCheck.Margin = new System.Windows.Forms.Padding(4);
            this.environCheck.Name = "environCheck";
            this.environCheck.Size = new System.Drawing.Size(149, 21);
            this.environCheck.TabIndex = 202;
            this.environCheck.Text = "Environment Limits";
            this.environCheck.UseVisualStyleBackColor = true;
            // 
            // numericUpDown2
            // 
            this.numericUpDown2.Location = new System.Drawing.Point(112, 76);
            this.numericUpDown2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.numericUpDown2.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDown2.Name = "numericUpDown2";
            this.numericUpDown2.Size = new System.Drawing.Size(64, 22);
            this.numericUpDown2.TabIndex = 150;
            // 
            // label154
            // 
            this.label154.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label154.Location = new System.Drawing.Point(7, 79);
            this.label154.Name = "label154";
            this.label154.Size = new System.Drawing.Size(88, 18);
            this.label154.TabIndex = 150;
            this.label154.Text = "Floor (A):";
            // 
            // BentoAdaptGripBox
            // 
            this.BentoAdaptGripBox.Controls.Add(this.BentoAdaptGripCheck);
            this.BentoAdaptGripBox.Controls.Add(this.label152);
            this.BentoAdaptGripBox.Controls.Add(this.BentoAdaptGripCtrl);
            this.BentoAdaptGripBox.Controls.Add(this.label151);
            this.BentoAdaptGripBox.Enabled = false;
            this.BentoAdaptGripBox.Location = new System.Drawing.Point(501, 6);
            this.BentoAdaptGripBox.Margin = new System.Windows.Forms.Padding(4);
            this.BentoAdaptGripBox.Name = "BentoAdaptGripBox";
            this.BentoAdaptGripBox.Padding = new System.Windows.Forms.Padding(4);
            this.BentoAdaptGripBox.Size = new System.Drawing.Size(303, 149);
            this.BentoAdaptGripBox.TabIndex = 208;
            this.BentoAdaptGripBox.TabStop = false;
            this.BentoAdaptGripBox.Text = "Grip Force Limit";
            // 
            // BentoAdaptGripCheck
            // 
            this.BentoAdaptGripCheck.AutoSize = true;
            this.BentoAdaptGripCheck.Location = new System.Drawing.Point(11, 25);
            this.BentoAdaptGripCheck.Margin = new System.Windows.Forms.Padding(4);
            this.BentoAdaptGripCheck.Name = "BentoAdaptGripCheck";
            this.BentoAdaptGripCheck.Size = new System.Drawing.Size(130, 21);
            this.BentoAdaptGripCheck.TabIndex = 202;
            this.BentoAdaptGripCheck.Text = "Grip Force Limit";
            this.BentoAdaptGripCheck.UseVisualStyleBackColor = true;
            // 
            // label152
            // 
            this.label152.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label152.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
            this.label152.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label152.Location = new System.Drawing.Point(7, 91);
            this.label152.Name = "label152";
            this.label152.Size = new System.Drawing.Size(289, 54);
            this.label152.TabIndex = 203;
            this.label152.Text = "Enable grip force limit and set to threshold \r\nof 400 or lower to help prevent ov" +
    "erloading\r\nof the hand servo.";
            // 
            // BentoAdaptGripCtrl
            // 
            this.BentoAdaptGripCtrl.Location = new System.Drawing.Point(184, 53);
            this.BentoAdaptGripCtrl.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.BentoAdaptGripCtrl.Maximum = new decimal(new int[] {
            1023,
            0,
            0,
            0});
            this.BentoAdaptGripCtrl.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.BentoAdaptGripCtrl.Name = "BentoAdaptGripCtrl";
            this.BentoAdaptGripCtrl.Size = new System.Drawing.Size(64, 22);
            this.BentoAdaptGripCtrl.TabIndex = 150;
            this.BentoAdaptGripCtrl.Value = new decimal(new int[] {
            400,
            0,
            0,
            0});
            // 
            // label151
            // 
            this.label151.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label151.Location = new System.Drawing.Point(7, 55);
            this.label151.Name = "label151";
            this.label151.Size = new System.Drawing.Size(172, 18);
            this.label151.TabIndex = 150;
            this.label151.Text = "Load Threshold (1-1023):";
            // 
            // tabXPC
            // 
            this.tabXPC.Controls.Add(this.VoiceCoilCommBox);
            this.tabXPC.Controls.Add(this.groupBox2);
            this.tabXPC.Controls.Add(this.LEDbox);
            this.tabXPC.Controls.Add(this.EMGParamBox);
            this.tabXPC.Controls.Add(this.RobotBox);
            this.tabXPC.Controls.Add(this.MLBox);
            this.tabXPC.Location = new System.Drawing.Point(4, 25);
            this.tabXPC.Margin = new System.Windows.Forms.Padding(4);
            this.tabXPC.Name = "tabXPC";
            this.tabXPC.Size = new System.Drawing.Size(1552, 793);
            this.tabXPC.TabIndex = 3;
            this.tabXPC.Text = "xPC Target";
            this.tabXPC.UseVisualStyleBackColor = true;
            // 
            // tabViz
            // 
            this.tabViz.Controls.Add(this.ArduinoInputGroupBox);
            this.tabViz.Controls.Add(this.biopatrecGroupBox);
            this.tabViz.Controls.Add(this.SLRTgroupBox);
            this.tabViz.Controls.Add(this.KBgroupBox);
            this.tabViz.Controls.Add(this.xBoxGroupBox);
            this.tabViz.Controls.Add(this.MYOgroupBox);
            this.tabViz.Location = new System.Drawing.Point(4, 25);
            this.tabViz.Margin = new System.Windows.Forms.Padding(4);
            this.tabViz.Name = "tabViz";
            this.tabViz.Size = new System.Drawing.Size(1552, 793);
            this.tabViz.TabIndex = 4;
            this.tabViz.Text = "Visualization";
            this.tabViz.UseVisualStyleBackColor = true;
            // 
            // ArduinoInputGroupBox
            // 
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A0);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A1);
            this.ArduinoInputGroupBox.Controls.Add(this.label207);
            this.ArduinoInputGroupBox.Controls.Add(this.label208);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A2);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A3);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A4);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A5);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A6);
            this.ArduinoInputGroupBox.Controls.Add(this.label214);
            this.ArduinoInputGroupBox.Controls.Add(this.arduino_A7);
            this.ArduinoInputGroupBox.Controls.Add(this.label216);
            this.ArduinoInputGroupBox.Controls.Add(this.label217);
            this.ArduinoInputGroupBox.Controls.Add(this.label218);
            this.ArduinoInputGroupBox.Controls.Add(this.label219);
            this.ArduinoInputGroupBox.Controls.Add(this.label220);
            this.ArduinoInputGroupBox.Enabled = false;
            this.ArduinoInputGroupBox.Location = new System.Drawing.Point(361, 282);
            this.ArduinoInputGroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.ArduinoInputGroupBox.Name = "ArduinoInputGroupBox";
            this.ArduinoInputGroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.ArduinoInputGroupBox.Size = new System.Drawing.Size(349, 202);
            this.ArduinoInputGroupBox.TabIndex = 211;
            this.ArduinoInputGroupBox.TabStop = false;
            this.ArduinoInputGroupBox.Text = "Arduino Analog Inputs";
            // 
            // arduino_A0
            // 
            this.arduino_A0.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A0.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A0.Location = new System.Drawing.Point(52, 20);
            this.arduino_A0.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A0.Name = "arduino_A0";
            this.arduino_A0.Size = new System.Drawing.Size(79, 19);
            this.arduino_A0.TabIndex = 185;
            this.arduino_A0.Text = "1.0";
            this.arduino_A0.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // arduino_A1
            // 
            this.arduino_A1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A1.Location = new System.Drawing.Point(52, 37);
            this.arduino_A1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A1.Name = "arduino_A1";
            this.arduino_A1.Size = new System.Drawing.Size(79, 19);
            this.arduino_A1.TabIndex = 186;
            this.arduino_A1.Text = "1.0";
            this.arduino_A1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label207
            // 
            this.label207.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label207.Location = new System.Drawing.Point(13, 37);
            this.label207.Name = "label207";
            this.label207.Size = new System.Drawing.Size(39, 18);
            this.label207.TabIndex = 188;
            this.label207.Text = "A1:";
            this.label207.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label208
            // 
            this.label208.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label208.Location = new System.Drawing.Point(3, 18);
            this.label208.Name = "label208";
            this.label208.Size = new System.Drawing.Size(49, 18);
            this.label208.TabIndex = 187;
            this.label208.Text = "A0:";
            this.label208.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // arduino_A2
            // 
            this.arduino_A2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A2.Location = new System.Drawing.Point(52, 64);
            this.arduino_A2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A2.Name = "arduino_A2";
            this.arduino_A2.Size = new System.Drawing.Size(79, 19);
            this.arduino_A2.TabIndex = 157;
            this.arduino_A2.Text = "1.0";
            this.arduino_A2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // arduino_A3
            // 
            this.arduino_A3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A3.Location = new System.Drawing.Point(52, 81);
            this.arduino_A3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A3.Name = "arduino_A3";
            this.arduino_A3.Size = new System.Drawing.Size(79, 19);
            this.arduino_A3.TabIndex = 158;
            this.arduino_A3.Text = "1.0";
            this.arduino_A3.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // arduino_A4
            // 
            this.arduino_A4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A4.Location = new System.Drawing.Point(52, 107);
            this.arduino_A4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A4.Name = "arduino_A4";
            this.arduino_A4.Size = new System.Drawing.Size(79, 19);
            this.arduino_A4.TabIndex = 159;
            this.arduino_A4.Text = "1.0";
            this.arduino_A4.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // arduino_A5
            // 
            this.arduino_A5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A5.Location = new System.Drawing.Point(52, 126);
            this.arduino_A5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A5.Name = "arduino_A5";
            this.arduino_A5.Size = new System.Drawing.Size(79, 18);
            this.arduino_A5.TabIndex = 160;
            this.arduino_A5.Text = "1.0";
            this.arduino_A5.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // arduino_A6
            // 
            this.arduino_A6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A6.Location = new System.Drawing.Point(52, 150);
            this.arduino_A6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A6.Name = "arduino_A6";
            this.arduino_A6.Size = new System.Drawing.Size(79, 18);
            this.arduino_A6.TabIndex = 161;
            this.arduino_A6.Text = "1.0";
            this.arduino_A6.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label214
            // 
            this.label214.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label214.Location = new System.Drawing.Point(7, 166);
            this.label214.Name = "label214";
            this.label214.Size = new System.Drawing.Size(44, 18);
            this.label214.TabIndex = 184;
            this.label214.Text = "A7:";
            this.label214.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // arduino_A7
            // 
            this.arduino_A7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.arduino_A7.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.arduino_A7.Location = new System.Drawing.Point(52, 167);
            this.arduino_A7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.arduino_A7.Name = "arduino_A7";
            this.arduino_A7.Size = new System.Drawing.Size(79, 18);
            this.arduino_A7.TabIndex = 162;
            this.arduino_A7.Text = "1.0";
            this.arduino_A7.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label216
            // 
            this.label216.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label216.Location = new System.Drawing.Point(7, 148);
            this.label216.Name = "label216";
            this.label216.Size = new System.Drawing.Size(44, 18);
            this.label216.TabIndex = 183;
            this.label216.Text = "A6:";
            this.label216.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label217
            // 
            this.label217.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label217.Location = new System.Drawing.Point(7, 126);
            this.label217.Name = "label217";
            this.label217.Size = new System.Drawing.Size(44, 18);
            this.label217.TabIndex = 182;
            this.label217.Text = "A5:";
            this.label217.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label218
            // 
            this.label218.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label218.Location = new System.Drawing.Point(13, 107);
            this.label218.Name = "label218";
            this.label218.Size = new System.Drawing.Size(39, 18);
            this.label218.TabIndex = 181;
            this.label218.Text = "A4:";
            this.label218.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label219
            // 
            this.label219.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label219.Location = new System.Drawing.Point(13, 81);
            this.label219.Name = "label219";
            this.label219.Size = new System.Drawing.Size(39, 18);
            this.label219.TabIndex = 180;
            this.label219.Text = "A3:";
            this.label219.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label220
            // 
            this.label220.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label220.Location = new System.Drawing.Point(13, 63);
            this.label220.Name = "label220";
            this.label220.Size = new System.Drawing.Size(39, 18);
            this.label220.TabIndex = 179;
            this.label220.Text = "A2:";
            this.label220.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // biopatrecGroupBox
            // 
            this.biopatrecGroupBox.Controls.Add(this.label184);
            this.biopatrecGroupBox.Controls.Add(this.label182);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass12);
            this.biopatrecGroupBox.Controls.Add(this.label165);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass24);
            this.biopatrecGroupBox.Controls.Add(this.label169);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass23);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass17);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass18);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass21);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass20);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass19);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass22);
            this.biopatrecGroupBox.Controls.Add(this.label170);
            this.biopatrecGroupBox.Controls.Add(this.label171);
            this.biopatrecGroupBox.Controls.Add(this.label172);
            this.biopatrecGroupBox.Controls.Add(this.label173);
            this.biopatrecGroupBox.Controls.Add(this.label175);
            this.biopatrecGroupBox.Controls.Add(this.label181);
            this.biopatrecGroupBox.Controls.Add(this.label201);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass11);
            this.biopatrecGroupBox.Controls.Add(this.label161);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass10);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass3);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass2);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass1);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass0);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass4);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass5);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass8);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass7);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass13);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass14);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass15);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass16);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass6);
            this.biopatrecGroupBox.Controls.Add(this.label183);
            this.biopatrecGroupBox.Controls.Add(this.label185);
            this.biopatrecGroupBox.Controls.Add(this.label187);
            this.biopatrecGroupBox.Controls.Add(this.label189);
            this.biopatrecGroupBox.Controls.Add(this.BPRclass9);
            this.biopatrecGroupBox.Controls.Add(this.label190);
            this.biopatrecGroupBox.Controls.Add(this.label191);
            this.biopatrecGroupBox.Controls.Add(this.label192);
            this.biopatrecGroupBox.Controls.Add(this.label193);
            this.biopatrecGroupBox.Controls.Add(this.label194);
            this.biopatrecGroupBox.Controls.Add(this.label195);
            this.biopatrecGroupBox.Controls.Add(this.label196);
            this.biopatrecGroupBox.Controls.Add(this.label197);
            this.biopatrecGroupBox.Controls.Add(this.label198);
            this.biopatrecGroupBox.Controls.Add(this.label199);
            this.biopatrecGroupBox.Enabled = false;
            this.biopatrecGroupBox.Location = new System.Drawing.Point(361, 4);
            this.biopatrecGroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.biopatrecGroupBox.Name = "biopatrecGroupBox";
            this.biopatrecGroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.biopatrecGroupBox.Size = new System.Drawing.Size(349, 271);
            this.biopatrecGroupBox.TabIndex = 211;
            this.biopatrecGroupBox.TabStop = false;
            this.biopatrecGroupBox.Text = "BioPatRec";
            // 
            // label184
            // 
            this.label184.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label184.Location = new System.Drawing.Point(184, 48);
            this.label184.Name = "label184";
            this.label184.Size = new System.Drawing.Size(108, 18);
            this.label184.TabIndex = 205;
            this.label184.Text = "Little Extend:";
            this.label184.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label182
            // 
            this.label182.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label182.Location = new System.Drawing.Point(7, 238);
            this.label182.Name = "label182";
            this.label182.Size = new System.Drawing.Size(113, 18);
            this.label182.TabIndex = 204;
            this.label182.Text = "Middle Flex:";
            this.label182.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass12
            // 
            this.BPRclass12.AutoSize = true;
            this.BPRclass12.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass12.Enabled = false;
            this.BPRclass12.Location = new System.Drawing.Point(120, 239);
            this.BPRclass12.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass12.Name = "BPRclass12";
            this.BPRclass12.Size = new System.Drawing.Size(18, 17);
            this.BPRclass12.TabIndex = 203;
            this.BPRclass12.UseVisualStyleBackColor = false;
            // 
            // label165
            // 
            this.label165.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label165.Location = new System.Drawing.Point(191, 218);
            this.label165.Name = "label165";
            this.label165.Size = new System.Drawing.Size(101, 18);
            this.label165.TabIndex = 202;
            this.label165.Text = "Side Grip:";
            this.label165.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass24
            // 
            this.BPRclass24.AutoSize = true;
            this.BPRclass24.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass24.Enabled = false;
            this.BPRclass24.Location = new System.Drawing.Point(293, 220);
            this.BPRclass24.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass24.Name = "BPRclass24";
            this.BPRclass24.Size = new System.Drawing.Size(18, 17);
            this.BPRclass24.TabIndex = 201;
            this.BPRclass24.UseVisualStyleBackColor = false;
            // 
            // label169
            // 
            this.label169.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label169.Location = new System.Drawing.Point(152, 199);
            this.label169.Name = "label169";
            this.label169.Size = new System.Drawing.Size(140, 18);
            this.label169.TabIndex = 196;
            this.label169.Text = "Rotate Elbow Ext:";
            this.label169.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass23
            // 
            this.BPRclass23.AutoSize = true;
            this.BPRclass23.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass23.Enabled = false;
            this.BPRclass23.Location = new System.Drawing.Point(293, 202);
            this.BPRclass23.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass23.Name = "BPRclass23";
            this.BPRclass23.Size = new System.Drawing.Size(18, 17);
            this.BPRclass23.TabIndex = 187;
            this.BPRclass23.UseVisualStyleBackColor = false;
            // 
            // BPRclass17
            // 
            this.BPRclass17.AutoSize = true;
            this.BPRclass17.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass17.Enabled = false;
            this.BPRclass17.Location = new System.Drawing.Point(293, 86);
            this.BPRclass17.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass17.Name = "BPRclass17";
            this.BPRclass17.Size = new System.Drawing.Size(18, 17);
            this.BPRclass17.TabIndex = 188;
            this.BPRclass17.UseVisualStyleBackColor = false;
            // 
            // BPRclass18
            // 
            this.BPRclass18.AutoSize = true;
            this.BPRclass18.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass18.Enabled = false;
            this.BPRclass18.Location = new System.Drawing.Point(293, 106);
            this.BPRclass18.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass18.Name = "BPRclass18";
            this.BPRclass18.Size = new System.Drawing.Size(18, 17);
            this.BPRclass18.TabIndex = 189;
            this.BPRclass18.UseVisualStyleBackColor = false;
            // 
            // BPRclass21
            // 
            this.BPRclass21.AutoSize = true;
            this.BPRclass21.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass21.Enabled = false;
            this.BPRclass21.Location = new System.Drawing.Point(293, 164);
            this.BPRclass21.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass21.Name = "BPRclass21";
            this.BPRclass21.Size = new System.Drawing.Size(18, 17);
            this.BPRclass21.TabIndex = 190;
            this.BPRclass21.UseVisualStyleBackColor = false;
            // 
            // BPRclass20
            // 
            this.BPRclass20.AutoSize = true;
            this.BPRclass20.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass20.Enabled = false;
            this.BPRclass20.Location = new System.Drawing.Point(293, 145);
            this.BPRclass20.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass20.Name = "BPRclass20";
            this.BPRclass20.Size = new System.Drawing.Size(18, 17);
            this.BPRclass20.TabIndex = 191;
            this.BPRclass20.UseVisualStyleBackColor = false;
            // 
            // BPRclass19
            // 
            this.BPRclass19.AutoSize = true;
            this.BPRclass19.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass19.Enabled = false;
            this.BPRclass19.Location = new System.Drawing.Point(293, 124);
            this.BPRclass19.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass19.Name = "BPRclass19";
            this.BPRclass19.Size = new System.Drawing.Size(18, 17);
            this.BPRclass19.TabIndex = 192;
            this.BPRclass19.UseVisualStyleBackColor = false;
            // 
            // BPRclass22
            // 
            this.BPRclass22.AutoSize = true;
            this.BPRclass22.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass22.CheckAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.BPRclass22.Enabled = false;
            this.BPRclass22.Location = new System.Drawing.Point(292, 183);
            this.BPRclass22.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass22.Name = "BPRclass22";
            this.BPRclass22.Size = new System.Drawing.Size(18, 17);
            this.BPRclass22.TabIndex = 200;
            this.BPRclass22.UseVisualStyleBackColor = false;
            // 
            // label170
            // 
            this.label170.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label170.Location = new System.Drawing.Point(184, 145);
            this.label170.Name = "label170";
            this.label170.Size = new System.Drawing.Size(108, 18);
            this.label170.TabIndex = 193;
            this.label170.Text = "Flex Elbow:";
            this.label170.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label171
            // 
            this.label171.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label171.Location = new System.Drawing.Point(151, 123);
            this.label171.Name = "label171";
            this.label171.Size = new System.Drawing.Size(141, 18);
            this.label171.TabIndex = 199;
            this.label171.Text = "Thumb Yaw Extend:";
            this.label171.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label172
            // 
            this.label172.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label172.Location = new System.Drawing.Point(191, 164);
            this.label172.Name = "label172";
            this.label172.Size = new System.Drawing.Size(101, 18);
            this.label172.TabIndex = 194;
            this.label172.Text = "Extend Elbow:";
            this.label172.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label173
            // 
            this.label173.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label173.Location = new System.Drawing.Point(171, 106);
            this.label173.Name = "label173";
            this.label173.Size = new System.Drawing.Size(121, 18);
            this.label173.TabIndex = 198;
            this.label173.Text = "Thumb Yaw Flex:";
            this.label173.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label175
            // 
            this.label175.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label175.Location = new System.Drawing.Point(173, 181);
            this.label175.Name = "label175";
            this.label175.Size = new System.Drawing.Size(119, 18);
            this.label175.TabIndex = 195;
            this.label175.Text = "Rotate Elbow Int:";
            this.label175.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label181
            // 
            this.label181.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label181.Location = new System.Drawing.Point(185, 86);
            this.label181.Name = "label181";
            this.label181.Size = new System.Drawing.Size(107, 18);
            this.label181.TabIndex = 197;
            this.label181.Text = "Point:";
            this.label181.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label201
            // 
            this.label201.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label201.Location = new System.Drawing.Point(7, 219);
            this.label201.Name = "label201";
            this.label201.Size = new System.Drawing.Size(113, 18);
            this.label201.TabIndex = 186;
            this.label201.Text = "Middle Extend:";
            this.label201.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass11
            // 
            this.BPRclass11.AutoSize = true;
            this.BPRclass11.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass11.Enabled = false;
            this.BPRclass11.Location = new System.Drawing.Point(120, 220);
            this.BPRclass11.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass11.Name = "BPRclass11";
            this.BPRclass11.Size = new System.Drawing.Size(18, 17);
            this.BPRclass11.TabIndex = 185;
            this.BPRclass11.UseVisualStyleBackColor = false;
            // 
            // label161
            // 
            this.label161.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label161.Location = new System.Drawing.Point(13, 201);
            this.label161.Name = "label161";
            this.label161.Size = new System.Drawing.Size(107, 18);
            this.label161.TabIndex = 170;
            this.label161.Text = "Index Flex:";
            this.label161.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass10
            // 
            this.BPRclass10.AutoSize = true;
            this.BPRclass10.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass10.Enabled = false;
            this.BPRclass10.Location = new System.Drawing.Point(120, 202);
            this.BPRclass10.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass10.Name = "BPRclass10";
            this.BPRclass10.Size = new System.Drawing.Size(18, 17);
            this.BPRclass10.TabIndex = 144;
            this.BPRclass10.UseVisualStyleBackColor = false;
            // 
            // BPRclass3
            // 
            this.BPRclass3.AutoSize = true;
            this.BPRclass3.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass3.Enabled = false;
            this.BPRclass3.Location = new System.Drawing.Point(120, 68);
            this.BPRclass3.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass3.Name = "BPRclass3";
            this.BPRclass3.Size = new System.Drawing.Size(18, 17);
            this.BPRclass3.TabIndex = 145;
            this.BPRclass3.UseVisualStyleBackColor = false;
            // 
            // BPRclass2
            // 
            this.BPRclass2.AutoSize = true;
            this.BPRclass2.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass2.Enabled = false;
            this.BPRclass2.Location = new System.Drawing.Point(120, 49);
            this.BPRclass2.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass2.Name = "BPRclass2";
            this.BPRclass2.Size = new System.Drawing.Size(18, 17);
            this.BPRclass2.TabIndex = 146;
            this.BPRclass2.UseVisualStyleBackColor = false;
            // 
            // BPRclass1
            // 
            this.BPRclass1.AutoSize = true;
            this.BPRclass1.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass1.Enabled = false;
            this.BPRclass1.Location = new System.Drawing.Point(120, 32);
            this.BPRclass1.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass1.Name = "BPRclass1";
            this.BPRclass1.Size = new System.Drawing.Size(18, 17);
            this.BPRclass1.TabIndex = 147;
            this.BPRclass1.UseVisualStyleBackColor = false;
            // 
            // BPRclass0
            // 
            this.BPRclass0.AutoSize = true;
            this.BPRclass0.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass0.Enabled = false;
            this.BPRclass0.Location = new System.Drawing.Point(120, 15);
            this.BPRclass0.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass0.Name = "BPRclass0";
            this.BPRclass0.Size = new System.Drawing.Size(18, 17);
            this.BPRclass0.TabIndex = 148;
            this.BPRclass0.UseVisualStyleBackColor = false;
            // 
            // BPRclass4
            // 
            this.BPRclass4.AutoSize = true;
            this.BPRclass4.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass4.Enabled = false;
            this.BPRclass4.Location = new System.Drawing.Point(120, 86);
            this.BPRclass4.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass4.Name = "BPRclass4";
            this.BPRclass4.Size = new System.Drawing.Size(18, 17);
            this.BPRclass4.TabIndex = 149;
            this.BPRclass4.UseVisualStyleBackColor = false;
            // 
            // BPRclass5
            // 
            this.BPRclass5.AutoSize = true;
            this.BPRclass5.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass5.Enabled = false;
            this.BPRclass5.Location = new System.Drawing.Point(120, 106);
            this.BPRclass5.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass5.Name = "BPRclass5";
            this.BPRclass5.Size = new System.Drawing.Size(18, 17);
            this.BPRclass5.TabIndex = 150;
            this.BPRclass5.UseVisualStyleBackColor = false;
            // 
            // BPRclass8
            // 
            this.BPRclass8.AutoSize = true;
            this.BPRclass8.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass8.Enabled = false;
            this.BPRclass8.Location = new System.Drawing.Point(120, 164);
            this.BPRclass8.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass8.Name = "BPRclass8";
            this.BPRclass8.Size = new System.Drawing.Size(18, 17);
            this.BPRclass8.TabIndex = 151;
            this.BPRclass8.UseVisualStyleBackColor = false;
            // 
            // BPRclass7
            // 
            this.BPRclass7.AutoSize = true;
            this.BPRclass7.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass7.Enabled = false;
            this.BPRclass7.Location = new System.Drawing.Point(120, 145);
            this.BPRclass7.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass7.Name = "BPRclass7";
            this.BPRclass7.Size = new System.Drawing.Size(18, 17);
            this.BPRclass7.TabIndex = 152;
            this.BPRclass7.UseVisualStyleBackColor = false;
            // 
            // BPRclass13
            // 
            this.BPRclass13.AutoSize = true;
            this.BPRclass13.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass13.Enabled = false;
            this.BPRclass13.Location = new System.Drawing.Point(293, 15);
            this.BPRclass13.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass13.Name = "BPRclass13";
            this.BPRclass13.Size = new System.Drawing.Size(18, 17);
            this.BPRclass13.TabIndex = 153;
            this.BPRclass13.UseVisualStyleBackColor = false;
            // 
            // BPRclass14
            // 
            this.BPRclass14.AutoSize = true;
            this.BPRclass14.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass14.Enabled = false;
            this.BPRclass14.Location = new System.Drawing.Point(293, 32);
            this.BPRclass14.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass14.Name = "BPRclass14";
            this.BPRclass14.Size = new System.Drawing.Size(18, 17);
            this.BPRclass14.TabIndex = 154;
            this.BPRclass14.UseVisualStyleBackColor = false;
            // 
            // BPRclass15
            // 
            this.BPRclass15.AutoSize = true;
            this.BPRclass15.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass15.Enabled = false;
            this.BPRclass15.Location = new System.Drawing.Point(293, 49);
            this.BPRclass15.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass15.Name = "BPRclass15";
            this.BPRclass15.Size = new System.Drawing.Size(18, 17);
            this.BPRclass15.TabIndex = 155;
            this.BPRclass15.UseVisualStyleBackColor = false;
            // 
            // BPRclass16
            // 
            this.BPRclass16.AutoSize = true;
            this.BPRclass16.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass16.Enabled = false;
            this.BPRclass16.Location = new System.Drawing.Point(293, 68);
            this.BPRclass16.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass16.Name = "BPRclass16";
            this.BPRclass16.Size = new System.Drawing.Size(18, 17);
            this.BPRclass16.TabIndex = 156;
            this.BPRclass16.UseVisualStyleBackColor = false;
            // 
            // BPRclass6
            // 
            this.BPRclass6.AutoSize = true;
            this.BPRclass6.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass6.Enabled = false;
            this.BPRclass6.Location = new System.Drawing.Point(120, 124);
            this.BPRclass6.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass6.Name = "BPRclass6";
            this.BPRclass6.Size = new System.Drawing.Size(18, 17);
            this.BPRclass6.TabIndex = 163;
            this.BPRclass6.UseVisualStyleBackColor = false;
            // 
            // label183
            // 
            this.label183.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label183.Location = new System.Drawing.Point(76, 12);
            this.label183.Name = "label183";
            this.label183.Size = new System.Drawing.Size(44, 18);
            this.label183.TabIndex = 150;
            this.label183.Text = "Rest:";
            this.label183.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label185
            // 
            this.label185.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label185.Location = new System.Drawing.Point(28, 31);
            this.label185.Name = "label185";
            this.label185.Size = new System.Drawing.Size(92, 18);
            this.label185.TabIndex = 164;
            this.label185.Text = "Open Hand:";
            this.label185.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label187
            // 
            this.label187.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label187.Location = new System.Drawing.Point(32, 49);
            this.label187.Name = "label187";
            this.label187.Size = new System.Drawing.Size(88, 18);
            this.label187.TabIndex = 165;
            this.label187.Text = "Close Hand:";
            this.label187.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label189
            // 
            this.label189.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label189.Location = new System.Drawing.Point(36, 66);
            this.label189.Name = "label189";
            this.label189.Size = new System.Drawing.Size(84, 18);
            this.label189.TabIndex = 166;
            this.label189.Text = "Flex Hand:";
            this.label189.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BPRclass9
            // 
            this.BPRclass9.AutoSize = true;
            this.BPRclass9.BackColor = System.Drawing.Color.Transparent;
            this.BPRclass9.CheckAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.BPRclass9.Enabled = false;
            this.BPRclass9.Location = new System.Drawing.Point(119, 183);
            this.BPRclass9.Margin = new System.Windows.Forms.Padding(4);
            this.BPRclass9.Name = "BPRclass9";
            this.BPRclass9.Size = new System.Drawing.Size(18, 17);
            this.BPRclass9.TabIndex = 178;
            this.BPRclass9.UseVisualStyleBackColor = false;
            // 
            // label190
            // 
            this.label190.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label190.Location = new System.Drawing.Point(12, 143);
            this.label190.Name = "label190";
            this.label190.Size = new System.Drawing.Size(108, 18);
            this.label190.TabIndex = 167;
            this.label190.Text = "Thumb Extend:";
            this.label190.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label191
            // 
            this.label191.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label191.Location = new System.Drawing.Point(27, 124);
            this.label191.Name = "label191";
            this.label191.Size = new System.Drawing.Size(93, 18);
            this.label191.TabIndex = 177;
            this.label191.Text = "Supination:";
            this.label191.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label192
            // 
            this.label192.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label192.Location = new System.Drawing.Point(32, 161);
            this.label192.Name = "label192";
            this.label192.Size = new System.Drawing.Size(88, 18);
            this.label192.TabIndex = 168;
            this.label192.Text = "Thumb Flex:";
            this.label192.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label193
            // 
            this.label193.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label193.Location = new System.Drawing.Point(32, 106);
            this.label193.Name = "label193";
            this.label193.Size = new System.Drawing.Size(88, 18);
            this.label193.TabIndex = 176;
            this.label193.Text = "Pronation:";
            this.label193.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label194
            // 
            this.label194.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label194.Location = new System.Drawing.Point(15, 181);
            this.label194.Name = "label194";
            this.label194.Size = new System.Drawing.Size(105, 18);
            this.label194.TabIndex = 169;
            this.label194.Text = "Index Extend:";
            this.label194.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label195
            // 
            this.label195.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label195.Location = new System.Drawing.Point(13, 86);
            this.label195.Name = "label195";
            this.label195.Size = new System.Drawing.Size(107, 18);
            this.label195.TabIndex = 175;
            this.label195.Text = "Extend Hand:";
            this.label195.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label196
            // 
            this.label196.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label196.Location = new System.Drawing.Point(172, 15);
            this.label196.Name = "label196";
            this.label196.Size = new System.Drawing.Size(120, 18);
            this.label196.TabIndex = 171;
            this.label196.Text = "Ring Extend:";
            this.label196.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label197
            // 
            this.label197.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label197.Location = new System.Drawing.Point(185, 68);
            this.label197.Name = "label197";
            this.label197.Size = new System.Drawing.Size(107, 18);
            this.label197.TabIndex = 174;
            this.label197.Text = "Little Flex:";
            this.label197.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label198
            // 
            this.label198.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label198.Location = new System.Drawing.Point(152, 31);
            this.label198.Name = "label198";
            this.label198.Size = new System.Drawing.Size(140, 18);
            this.label198.TabIndex = 172;
            this.label198.Text = "Ring Flex:";
            this.label198.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label199
            // 
            this.label199.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label199.Location = new System.Drawing.Point(144, 47);
            this.label199.Name = "label199";
            this.label199.Size = new System.Drawing.Size(107, 18);
            this.label199.TabIndex = 173;
            this.label199.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // SLRTgroupBox
            // 
            this.SLRTgroupBox.Controls.Add(this.slrt_ch1);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch2);
            this.SLRTgroupBox.Controls.Add(this.label167);
            this.SLRTgroupBox.Controls.Add(this.label168);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch3);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch4);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch5);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch6);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch7);
            this.SLRTgroupBox.Controls.Add(this.label174);
            this.SLRTgroupBox.Controls.Add(this.slrt_ch8);
            this.SLRTgroupBox.Controls.Add(this.label176);
            this.SLRTgroupBox.Controls.Add(this.label177);
            this.SLRTgroupBox.Controls.Add(this.label178);
            this.SLRTgroupBox.Controls.Add(this.label179);
            this.SLRTgroupBox.Controls.Add(this.label180);
            this.SLRTgroupBox.Enabled = false;
            this.SLRTgroupBox.Location = new System.Drawing.Point(361, 491);
            this.SLRTgroupBox.Margin = new System.Windows.Forms.Padding(4);
            this.SLRTgroupBox.Name = "SLRTgroupBox";
            this.SLRTgroupBox.Padding = new System.Windows.Forms.Padding(4);
            this.SLRTgroupBox.Size = new System.Drawing.Size(349, 202);
            this.SLRTgroupBox.TabIndex = 210;
            this.SLRTgroupBox.TabStop = false;
            this.SLRTgroupBox.Text = "Simulink Realtime";
            // 
            // slrt_ch1
            // 
            this.slrt_ch1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch1.Location = new System.Drawing.Point(52, 20);
            this.slrt_ch1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch1.Name = "slrt_ch1";
            this.slrt_ch1.Size = new System.Drawing.Size(79, 19);
            this.slrt_ch1.TabIndex = 185;
            this.slrt_ch1.Text = "1.0";
            this.slrt_ch1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // slrt_ch2
            // 
            this.slrt_ch2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch2.Location = new System.Drawing.Point(52, 37);
            this.slrt_ch2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch2.Name = "slrt_ch2";
            this.slrt_ch2.Size = new System.Drawing.Size(79, 19);
            this.slrt_ch2.TabIndex = 186;
            this.slrt_ch2.Text = "1.0";
            this.slrt_ch2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label167
            // 
            this.label167.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label167.Location = new System.Drawing.Point(13, 37);
            this.label167.Name = "label167";
            this.label167.Size = new System.Drawing.Size(39, 18);
            this.label167.TabIndex = 188;
            this.label167.Text = "Ch2:";
            this.label167.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label168
            // 
            this.label168.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label168.Location = new System.Drawing.Point(3, 18);
            this.label168.Name = "label168";
            this.label168.Size = new System.Drawing.Size(49, 18);
            this.label168.TabIndex = 187;
            this.label168.Text = "Ch1:";
            this.label168.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // slrt_ch3
            // 
            this.slrt_ch3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch3.Location = new System.Drawing.Point(52, 64);
            this.slrt_ch3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch3.Name = "slrt_ch3";
            this.slrt_ch3.Size = new System.Drawing.Size(79, 19);
            this.slrt_ch3.TabIndex = 157;
            this.slrt_ch3.Text = "1.0";
            this.slrt_ch3.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // slrt_ch4
            // 
            this.slrt_ch4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch4.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch4.Location = new System.Drawing.Point(52, 81);
            this.slrt_ch4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch4.Name = "slrt_ch4";
            this.slrt_ch4.Size = new System.Drawing.Size(79, 19);
            this.slrt_ch4.TabIndex = 158;
            this.slrt_ch4.Text = "1.0";
            this.slrt_ch4.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // slrt_ch5
            // 
            this.slrt_ch5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch5.Location = new System.Drawing.Point(52, 107);
            this.slrt_ch5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch5.Name = "slrt_ch5";
            this.slrt_ch5.Size = new System.Drawing.Size(79, 19);
            this.slrt_ch5.TabIndex = 159;
            this.slrt_ch5.Text = "1.0";
            this.slrt_ch5.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // slrt_ch6
            // 
            this.slrt_ch6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch6.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch6.Location = new System.Drawing.Point(52, 126);
            this.slrt_ch6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch6.Name = "slrt_ch6";
            this.slrt_ch6.Size = new System.Drawing.Size(79, 18);
            this.slrt_ch6.TabIndex = 160;
            this.slrt_ch6.Text = "1.0";
            this.slrt_ch6.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // slrt_ch7
            // 
            this.slrt_ch7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch7.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch7.Location = new System.Drawing.Point(52, 150);
            this.slrt_ch7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch7.Name = "slrt_ch7";
            this.slrt_ch7.Size = new System.Drawing.Size(79, 18);
            this.slrt_ch7.TabIndex = 161;
            this.slrt_ch7.Text = "1.0";
            this.slrt_ch7.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label174
            // 
            this.label174.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label174.Location = new System.Drawing.Point(7, 166);
            this.label174.Name = "label174";
            this.label174.Size = new System.Drawing.Size(44, 18);
            this.label174.TabIndex = 184;
            this.label174.Text = "Ch8:";
            this.label174.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // slrt_ch8
            // 
            this.slrt_ch8.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.slrt_ch8.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.slrt_ch8.Location = new System.Drawing.Point(52, 167);
            this.slrt_ch8.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.slrt_ch8.Name = "slrt_ch8";
            this.slrt_ch8.Size = new System.Drawing.Size(79, 18);
            this.slrt_ch8.TabIndex = 162;
            this.slrt_ch8.Text = "1.0";
            this.slrt_ch8.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label176
            // 
            this.label176.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label176.Location = new System.Drawing.Point(7, 148);
            this.label176.Name = "label176";
            this.label176.Size = new System.Drawing.Size(44, 18);
            this.label176.TabIndex = 183;
            this.label176.Text = "Ch7:";
            this.label176.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label177
            // 
            this.label177.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label177.Location = new System.Drawing.Point(7, 126);
            this.label177.Name = "label177";
            this.label177.Size = new System.Drawing.Size(44, 18);
            this.label177.TabIndex = 182;
            this.label177.Text = "Ch6:";
            this.label177.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label178
            // 
            this.label178.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label178.Location = new System.Drawing.Point(13, 107);
            this.label178.Name = "label178";
            this.label178.Size = new System.Drawing.Size(39, 18);
            this.label178.TabIndex = 181;
            this.label178.Text = "Ch5:";
            this.label178.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label179
            // 
            this.label179.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label179.Location = new System.Drawing.Point(13, 81);
            this.label179.Name = "label179";
            this.label179.Size = new System.Drawing.Size(39, 18);
            this.label179.TabIndex = 180;
            this.label179.Text = "Ch4:";
            this.label179.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // label180
            // 
            this.label180.ImeMode = System.Windows.Forms.ImeMode.NoControl;
            this.label180.Location = new System.Drawing.Point(13, 63);
            this.label180.Name = "label180";
            this.label180.Size = new System.Drawing.Size(39, 18);
            this.label180.TabIndex = 179;
            this.label180.Text = "Ch3:";
            this.label180.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // statusPanel1
            // 
            this.statusPanel1.Controls.Add(this.MYOstatus);
            this.statusPanel1.Controls.Add(this.BentoErrorText);
            this.statusPanel1.Controls.Add(this.label164);
            this.statusPanel1.Controls.Add(this.BentoErrorColor);
            this.statusPanel1.Controls.Add(this.BentoRunStatus);
            this.statusPanel1.Controls.Add(this.label117);
            this.statusPanel1.Controls.Add(this.delay);
            this.statusPanel1.Controls.Add(this.BentoStatus);
            this.statusPanel1.Controls.Add(this.label149);
            this.statusPanel1.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.statusPanel1.Location = new System.Drawing.Point(0, 810);
            this.statusPanel1.Margin = new System.Windows.Forms.Padding(4);
            this.statusPanel1.Name = "statusPanel1";
            this.statusPanel1.Size = new System.Drawing.Size(1576, 27);
            this.statusPanel1.TabIndex = 223;
            // 
            // MYOstatus
            // 
            this.MYOstatus.AutoSize = true;
            this.MYOstatus.Location = new System.Drawing.Point(661, 5);
            this.MYOstatus.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.MYOstatus.Name = "MYOstatus";
            this.MYOstatus.Size = new System.Drawing.Size(94, 17);
            this.MYOstatus.TabIndex = 226;
            this.MYOstatus.Text = "Disconnected";
            // 
            // BentoErrorText
            // 
            this.BentoErrorText.AutoSize = true;
            this.BentoErrorText.Location = new System.Drawing.Point(420, 5);
            this.BentoErrorText.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.BentoErrorText.Name = "BentoErrorText";
            this.BentoErrorText.Size = new System.Drawing.Size(0, 17);
            this.BentoErrorText.TabIndex = 225;
            // 
            // label164
            // 
            this.label164.AutoSize = true;
            this.label164.Location = new System.Drawing.Point(573, 5);
            this.label164.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label164.Name = "label164";
            this.label164.Size = new System.Drawing.Size(87, 17);
            this.label164.TabIndex = 227;
            this.label164.Text = "MYO Status:";
            // 
            // BentoErrorColor
            // 
            this.BentoErrorColor.ForeColor = System.Drawing.Color.Firebrick;
            this.BentoErrorColor.Location = new System.Drawing.Point(353, 5);
            this.BentoErrorColor.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.BentoErrorColor.Name = "BentoErrorColor";
            this.BentoErrorColor.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.BentoErrorColor.Size = new System.Drawing.Size(75, 16);
            this.BentoErrorColor.TabIndex = 225;
            this.BentoErrorColor.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // BentoRunStatus
            // 
            this.BentoRunStatus.Enabled = false;
            this.BentoRunStatus.Location = new System.Drawing.Point(269, 0);
            this.BentoRunStatus.Margin = new System.Windows.Forms.Padding(4);
            this.BentoRunStatus.Name = "BentoRunStatus";
            this.BentoRunStatus.Size = new System.Drawing.Size(76, 25);
            this.BentoRunStatus.TabIndex = 223;
            this.BentoRunStatus.Text = "Suspend";
            this.BentoRunStatus.UseVisualStyleBackColor = true;
            this.BentoRunStatus.Click += new System.EventHandler(this.BentoRunStatus_Click);
            // 
            // BentoStatus
            // 
            this.BentoStatus.AutoSize = true;
            this.BentoStatus.Location = new System.Drawing.Point(99, 5);
            this.BentoStatus.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.BentoStatus.Name = "BentoStatus";
            this.BentoStatus.Size = new System.Drawing.Size(94, 17);
            this.BentoStatus.TabIndex = 224;
            this.BentoStatus.Text = "Disconnected";
            // 
            // label149
            // 
            this.label149.AutoSize = true;
            this.label149.Location = new System.Drawing.Point(5, 5);
            this.label149.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label149.Name = "label149";
            this.label149.Size = new System.Drawing.Size(93, 17);
            this.label149.TabIndex = 224;
            this.label149.Text = "Bento Status:";
            // 
            // serialArduinoInput
            // 
            this.serialArduinoInput.RtsEnable = true;
            this.serialArduinoInput.DataReceived += new System.IO.Ports.SerialDataReceivedEventHandler(this.serialArduinoInput_DataReceived);
            // 
            // mainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoScroll = true;
            this.ClientSize = new System.Drawing.Size(1576, 837);
            this.Controls.Add(this.statusPanel1);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.MenuStrip1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.KeyPreview = true;
            this.Margin = new System.Windows.Forms.Padding(4);
            this.MaximumSize = new System.Drawing.Size(1594, 884);
            this.Name = "mainForm";
            this.Text = "brachI/Oplexus - V1.0";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.mainForm_FormClosing);
            this.Load += new System.EventHandler(this.mainForm_Load);
            this.KeyDown += new System.Windows.Forms.KeyEventHandler(this.mainForm_KeyDown);
            this.MenuStrip1.ResumeLayout(false);
            this.MenuStrip1.PerformLayout();
            this.VoiceCoilCommBox.ResumeLayout(false);
            this.VoiceCoilCommBox.PerformLayout();
            this.EMGParamBox.ResumeLayout(false);
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox8)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch8_gain_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch7_gain_ctrl)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch6_gain_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch5_gain_ctrl)).EndInit();
            this.DoF2box.ResumeLayout(false);
            this.DoF2box.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch4_gain_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch3_gain_ctrl)).EndInit();
            this.DoF1box.ResumeLayout(false);
            this.DoF1box.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch2_gain_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch1_gain_ctrl)).EndInit();
            this.SwitchBox.ResumeLayout(false);
            this.SwitchBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.cctime_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_smax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_smin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ch9_gain_ctrl)).EndInit();
            this.RobotBox.ResumeLayout(false);
            this.RobotBox.PerformLayout();
            this.RobotFeedbackBox.ResumeLayout(false);
            this.RobotFeedbackBox.PerformLayout();
            this.RobotParamBox.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.hand_wmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_wmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_pmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_pmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_wmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_wmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_pmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_pmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_wmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_wmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_pmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_pmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_wmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_wmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_pmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_pmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_wmax_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_wmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_pmin_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_pmax_ctrl)).EndInit();
            this.SimBox.ResumeLayout(false);
            this.SimBox.PerformLayout();
            this.LEDbox.ResumeLayout(false);
            this.LEDbox.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.MLBox.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.hand_w)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.hand_p)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_w)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristRot_p)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_w)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.elbow_p)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_w)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.shoulder_p)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_w)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.wristFlex_p)).EndInit();
            this.BentoGroupBox.ResumeLayout(false);
            this.xBoxGroupBox.ResumeLayout(false);
            this.xBoxGroupBox.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.groupBox18.ResumeLayout(false);
            this.groupBox18.PerformLayout();
            this.groupBox17.ResumeLayout(false);
            this.groupBox15.ResumeLayout(false);
            this.groupBox15.PerformLayout();
            this.groupBox7.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox10)).EndInit();
            this.groupBox8.ResumeLayout(false);
            this.groupBox8.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox11)).EndInit();
            this.groupBox5.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox9)).EndInit();
            this.groupBox6.ResumeLayout(false);
            this.groupBox6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox12)).EndInit();
            this.MYOgroupBox.ResumeLayout(false);
            this.KBgroupBox.ResumeLayout(false);
            this.KBgroupBox.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabIO.ResumeLayout(false);
            this.tabIO.PerformLayout();
            this.groupBox9.ResumeLayout(false);
            this.tabMapping.ResumeLayout(false);
            this.LoggingGroupBox.ResumeLayout(false);
            this.LoggingGroupBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.log_number)).EndInit();
            this.groupBox16.ResumeLayout(false);
            this.groupBox16.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.switchSmaxCtrl2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchSminCtrl2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchGainCtrl2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchTimeCtrl2)).EndInit();
            this.groupBox11.ResumeLayout(false);
            this.groupBox14.ResumeLayout(false);
            this.groupBox14.PerformLayout();
            this.groupBox13.ResumeLayout(false);
            this.groupBox13.PerformLayout();
            this.groupBox12.ResumeLayout(false);
            this.groupBox12.PerformLayout();
            this.groupBox10.ResumeLayout(false);
            this.groupBox10.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.switchSmaxCtrl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchSminCtrl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchGainCtrl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.switchTimeCtrl1)).EndInit();
            this.tabBento.ResumeLayout(false);
            this.tabBento.PerformLayout();
            this.AutoLevellingBox.ResumeLayout(false);
            this.AutoLevellingBox.PerformLayout();
            this.FlexionPIDBox.ResumeLayout(false);
            this.FlexionPIDBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Kd_theta_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Ki_theta_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Kp_theta_ctrl)).EndInit();
            this.RotationPIDBox.ResumeLayout(false);
            this.RotationPIDBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Kd_phi_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Ki_phi_ctrl)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Kp_phi_ctrl)).EndInit();
            this.groupBox19.ResumeLayout(false);
            this.BentoEnvLimitsBox.ResumeLayout(false);
            this.BentoEnvLimitsBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown2)).EndInit();
            this.BentoAdaptGripBox.ResumeLayout(false);
            this.BentoAdaptGripBox.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.BentoAdaptGripCtrl)).EndInit();
            this.tabXPC.ResumeLayout(false);
            this.tabViz.ResumeLayout(false);
            this.ArduinoInputGroupBox.ResumeLayout(false);
            this.biopatrecGroupBox.ResumeLayout(false);
            this.biopatrecGroupBox.PerformLayout();
            this.SLRTgroupBox.ResumeLayout(false);
            this.statusPanel1.ResumeLayout(false);
            this.statusPanel1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private MathWorks.xPCTarget.FrameWork.xPCTargetPC tg;
        internal System.Windows.Forms.GroupBox VoiceCoilCommBox;
        internal System.Windows.Forms.Button loadDLMButton;
        internal System.Windows.Forms.Label model_name;
        internal System.Windows.Forms.Button unloadButton;
        internal System.Windows.Forms.Label Label3;
        internal System.Windows.Forms.Button startButton;
        internal System.Windows.Forms.Button loadButton;
        internal System.Windows.Forms.Button stopButton;
        internal System.Windows.Forms.Button disconnectButton;
        internal System.Windows.Forms.Label Label9;
        internal System.Windows.Forms.TextBox ipportTB;
        internal System.Windows.Forms.TextBox ipaddressTB;
        internal System.Windows.Forms.Label Label10;
        internal System.Windows.Forms.Button connectButton;
        internal System.Windows.Forms.MenuStrip MenuStrip1;
        internal System.Windows.Forms.ToolStripMenuItem FileToolStripMenuItem;
        internal System.Windows.Forms.ToolStripMenuItem NewToolStripMenuItem;
        internal System.Windows.Forms.ToolStripMenuItem OpenToolStripMenuItem;
        internal System.Windows.Forms.ToolStripSeparator toolStripSeparator;
        internal System.Windows.Forms.ToolStripMenuItem SaveAsToolStripMenuItem;
        internal System.Windows.Forms.ToolStripMenuItem ToolStripMenuItem1;
        internal System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        internal System.Windows.Forms.ToolStripMenuItem ExitToolStripMenuItem;
        internal System.Windows.Forms.ToolStripMenuItem HelpToolStripMenuItem;
        internal System.Windows.Forms.ToolStripMenuItem ContentsToolStripMenuItem;
        internal System.Windows.Forms.ToolStripSeparator toolStripSeparator5;
        internal System.Windows.Forms.ToolStripMenuItem AboutToolStripMenuItem;
        private System.IO.Ports.SerialPort serialPort1;
        internal System.Windows.Forms.GroupBox EMGParamBox;
        internal System.Windows.Forms.GroupBox DoF1box;
        internal System.Windows.Forms.Label Label41;
        internal System.Windows.Forms.ComboBox DoF1_mode_box;
        internal System.Windows.Forms.Label Label56;
        internal System.Windows.Forms.Label ch2_smax_label;
        internal System.Windows.Forms.Label ch2_smin_label;
        internal System.Windows.Forms.Label ch1_smax_label;
        internal System.Windows.Forms.Label ch1_smin_label;
        internal System.Windows.Forms.Label ch2_smin_tick;
        internal System.Windows.Forms.Label ch2_smax_tick;
        internal System.Windows.Forms.Label ch1_smin_tick;
        internal System.Windows.Forms.Label ch1_smax_tick;
        internal System.Windows.Forms.NumericUpDown ch2_smax_ctrl;
        internal System.Windows.Forms.NumericUpDown ch1_smax_ctrl;
        internal System.Windows.Forms.Label Label12;
        internal System.Windows.Forms.NumericUpDown ch2_smin_ctrl;
        internal System.Windows.Forms.NumericUpDown ch1_smin_ctrl;
        internal System.Windows.Forms.Label label1;
        internal System.Windows.Forms.Label Label8;
        internal System.Windows.Forms.NumericUpDown ch2_gain_ctrl;
        internal System.Windows.Forms.Label label4;
        internal System.Windows.Forms.Label label13;
        internal System.Windows.Forms.ProgressBar MAV1_bar;
        internal System.Windows.Forms.ComboBox DoF1_mapping_combobox;
        internal System.Windows.Forms.Label label14;
        internal System.Windows.Forms.Label label15;
        internal System.Windows.Forms.ProgressBar MAV2_bar;
        internal System.Windows.Forms.NumericUpDown ch1_gain_ctrl;
        internal System.Windows.Forms.Label label16;
        internal System.Windows.Forms.GroupBox SwitchBox;
        internal System.Windows.Forms.ComboBox Switch_cycle5_combobox;
        internal System.Windows.Forms.ComboBox Switch_cycle4_combobox;
        internal System.Windows.Forms.ComboBox Switch_cycle3_combobox;
        internal System.Windows.Forms.ComboBox Switch_cycle2_combobox;
        internal System.Windows.Forms.ComboBox Switch_cycle1_combobox;
        internal System.Windows.Forms.Label cycle_number;
        internal System.Windows.Forms.Label label17;
        internal System.Windows.Forms.ComboBox switch_dof_combobox;
        internal System.Windows.Forms.Label Label75;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.CheckBox DoF1_flip_checkBox;
        internal System.Windows.Forms.Label label25;
        private System.Windows.Forms.PictureBox pictureBox2;
        internal System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.PictureBox pictureBox7;
        internal System.Windows.Forms.Label label78;
        internal System.Windows.Forms.ComboBox DoF4_mode_box;
        private System.Windows.Forms.PictureBox pictureBox8;
        internal System.Windows.Forms.Label label79;
        internal System.Windows.Forms.Label ch8_smax_label;
        private System.Windows.Forms.CheckBox DoF4_flip_checkBox;
        internal System.Windows.Forms.Label ch8_smin_label;
        internal System.Windows.Forms.Label ch7_smax_label;
        internal System.Windows.Forms.Label label83;
        internal System.Windows.Forms.Label ch7_smin_label;
        internal System.Windows.Forms.Label ch8_smin_tick;
        internal System.Windows.Forms.Label ch8_smax_tick;
        internal System.Windows.Forms.Label ch7_smin_tick;
        internal System.Windows.Forms.Label ch7_smax_tick;
        internal System.Windows.Forms.NumericUpDown ch8_smax_ctrl;
        internal System.Windows.Forms.NumericUpDown ch7_smax_ctrl;
        internal System.Windows.Forms.Label label89;
        internal System.Windows.Forms.NumericUpDown ch8_smin_ctrl;
        internal System.Windows.Forms.NumericUpDown ch7_smin_ctrl;
        internal System.Windows.Forms.Label label90;
        internal System.Windows.Forms.Label label91;
        internal System.Windows.Forms.NumericUpDown ch8_gain_ctrl;
        internal System.Windows.Forms.Label label92;
        internal System.Windows.Forms.Label label93;
        internal System.Windows.Forms.ProgressBar MAV7_bar;
        internal System.Windows.Forms.ComboBox DoF4_mapping_combobox;
        internal System.Windows.Forms.Label label94;
        internal System.Windows.Forms.Label label95;
        internal System.Windows.Forms.ProgressBar MAV8_bar;
        internal System.Windows.Forms.NumericUpDown ch7_gain_ctrl;
        internal System.Windows.Forms.Label label96;
        internal System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.PictureBox pictureBox5;
        internal System.Windows.Forms.Label label51;
        internal System.Windows.Forms.ComboBox DoF3_mode_box;
        private System.Windows.Forms.PictureBox pictureBox6;
        internal System.Windows.Forms.Label label52;
        internal System.Windows.Forms.Label ch6_smax_label;
        private System.Windows.Forms.CheckBox DoF3_flip_checkBox;
        internal System.Windows.Forms.Label ch6_smin_label;
        internal System.Windows.Forms.Label ch5_smax_label;
        internal System.Windows.Forms.Label label57;
        internal System.Windows.Forms.Label ch5_smin_label;
        internal System.Windows.Forms.Label ch6_smin_tick;
        internal System.Windows.Forms.Label ch6_smax_tick;
        internal System.Windows.Forms.Label ch5_smin_tick;
        internal System.Windows.Forms.Label ch5_smax_tick;
        internal System.Windows.Forms.NumericUpDown ch6_smax_ctrl;
        internal System.Windows.Forms.NumericUpDown ch5_smax_ctrl;
        internal System.Windows.Forms.Label label63;
        internal System.Windows.Forms.NumericUpDown ch6_smin_ctrl;
        internal System.Windows.Forms.NumericUpDown ch5_smin_ctrl;
        internal System.Windows.Forms.Label label64;
        internal System.Windows.Forms.Label label65;
        internal System.Windows.Forms.NumericUpDown ch6_gain_ctrl;
        internal System.Windows.Forms.Label label66;
        internal System.Windows.Forms.Label label67;
        internal System.Windows.Forms.ProgressBar MAV5_bar;
        internal System.Windows.Forms.ComboBox DoF3_mapping_combobox;
        internal System.Windows.Forms.Label label68;
        internal System.Windows.Forms.Label label72;
        internal System.Windows.Forms.ProgressBar MAV6_bar;
        internal System.Windows.Forms.NumericUpDown ch5_gain_ctrl;
        internal System.Windows.Forms.Label label77;
        internal System.Windows.Forms.GroupBox DoF2box;
        private System.Windows.Forms.PictureBox pictureBox3;
        internal System.Windows.Forms.Label label26;
        internal System.Windows.Forms.ComboBox DoF2_mode_box;
        private System.Windows.Forms.PictureBox pictureBox4;
        internal System.Windows.Forms.Label label28;
        internal System.Windows.Forms.Label ch4_smax_label;
        private System.Windows.Forms.CheckBox DoF2_flip_checkBox;
        internal System.Windows.Forms.Label ch4_smin_label;
        internal System.Windows.Forms.Label ch3_smax_label;
        internal System.Windows.Forms.Label label33;
        internal System.Windows.Forms.Label ch3_smin_label;
        internal System.Windows.Forms.Label ch4_smin_tick;
        internal System.Windows.Forms.Label ch4_smax_tick;
        internal System.Windows.Forms.Label ch3_smin_tick;
        internal System.Windows.Forms.Label ch3_smax_tick;
        internal System.Windows.Forms.NumericUpDown ch4_smax_ctrl;
        internal System.Windows.Forms.NumericUpDown ch3_smax_ctrl;
        internal System.Windows.Forms.Label label43;
        internal System.Windows.Forms.NumericUpDown ch4_smin_ctrl;
        internal System.Windows.Forms.NumericUpDown ch3_smin_ctrl;
        internal System.Windows.Forms.Label label44;
        internal System.Windows.Forms.Label label45;
        internal System.Windows.Forms.NumericUpDown ch4_gain_ctrl;
        internal System.Windows.Forms.Label label46;
        internal System.Windows.Forms.Label label47;
        internal System.Windows.Forms.ProgressBar MAV3_bar;
        internal System.Windows.Forms.ComboBox DoF2_mapping_combobox;
        internal System.Windows.Forms.Label label48;
        internal System.Windows.Forms.Label label49;
        internal System.Windows.Forms.ProgressBar MAV4_bar;
        internal System.Windows.Forms.NumericUpDown ch3_gain_ctrl;
        internal System.Windows.Forms.Label label50;
        internal System.Windows.Forms.Label label31;
        internal System.Windows.Forms.ComboBox switch_mode_combobox;
        private System.Windows.Forms.CheckBox led_checkBox;
        private System.Windows.Forms.CheckBox vocal_checkBox;
        private System.Windows.Forms.CheckBox ding_checkBox;
        internal System.Windows.Forms.Label label102;
        private System.Windows.Forms.CheckBox cycle5_flip_checkBox;
        private System.Windows.Forms.CheckBox cycle4_flip_checkBox;
        private System.Windows.Forms.CheckBox cycle3_flip_checkBox;
        private System.Windows.Forms.CheckBox cycle2_flip_checkBox;
        private System.Windows.Forms.CheckBox cycle1_flip_checkBox;
        internal System.Windows.Forms.Label label74;
        internal System.Windows.Forms.Label label70;
        internal System.Windows.Forms.Label ch9_smax_label;
        internal System.Windows.Forms.Label ch9_smin_label;
        internal System.Windows.Forms.Label ch9_smin_tick;
        internal System.Windows.Forms.Label ch9_smax_tick;
        internal System.Windows.Forms.NumericUpDown ch9_smax_ctrl;
        internal System.Windows.Forms.Label label97;
        internal System.Windows.Forms.NumericUpDown ch9_smin_ctrl;
        internal System.Windows.Forms.Label label98;
        internal System.Windows.Forms.Label label99;
        internal System.Windows.Forms.Label label100;
        internal System.Windows.Forms.ProgressBar MAV9_bar;
        internal System.Windows.Forms.NumericUpDown ch9_gain_ctrl;
        internal System.Windows.Forms.Label label101;
        private System.Windows.Forms.CheckBox text_checkBox;
        internal System.Windows.Forms.GroupBox RobotBox;
        internal System.Windows.Forms.ComboBox hand_comboBox;
        internal System.Windows.Forms.Label label23;
        internal System.Windows.Forms.Button AX12stopBTN;
        internal System.Windows.Forms.Button AX12startBTN;
        private System.Windows.Forms.GroupBox RobotParamBox;
        internal System.Windows.Forms.NumericUpDown hand_wmax_ctrl;
        internal System.Windows.Forms.NumericUpDown hand_wmin_ctrl;
        internal System.Windows.Forms.NumericUpDown hand_pmin_ctrl;
        internal System.Windows.Forms.NumericUpDown hand_pmax_ctrl;
        internal System.Windows.Forms.Label label7;
        internal System.Windows.Forms.NumericUpDown wristRot_wmax_ctrl;
        internal System.Windows.Forms.NumericUpDown wristRot_wmin_ctrl;
        internal System.Windows.Forms.Label Label18;
        internal System.Windows.Forms.NumericUpDown wristRot_pmin_ctrl;
        internal System.Windows.Forms.NumericUpDown wristRot_pmax_ctrl;
        internal System.Windows.Forms.NumericUpDown elbow_wmax_ctrl;
        internal System.Windows.Forms.NumericUpDown elbow_wmin_ctrl;
        internal System.Windows.Forms.NumericUpDown elbow_pmin_ctrl;
        internal System.Windows.Forms.NumericUpDown elbow_pmax_ctrl;
        internal System.Windows.Forms.Label Label20;
        internal System.Windows.Forms.NumericUpDown shoulder_wmax_ctrl;
        internal System.Windows.Forms.NumericUpDown shoulder_wmin_ctrl;
        internal System.Windows.Forms.NumericUpDown shoulder_pmin_ctrl;
        internal System.Windows.Forms.NumericUpDown shoulder_pmax_ctrl;
        internal System.Windows.Forms.Label Label21;
        internal System.Windows.Forms.Label Label19;
        internal System.Windows.Forms.NumericUpDown wristFlex_wmax_ctrl;
        internal System.Windows.Forms.NumericUpDown wristFlex_wmin_ctrl;
        internal System.Windows.Forms.NumericUpDown wristFlex_pmin_ctrl;
        internal System.Windows.Forms.NumericUpDown wristFlex_pmax_ctrl;
        internal System.Windows.Forms.Label label5;
        internal System.Windows.Forms.Label label6;
        internal System.Windows.Forms.Label label11;
        internal System.Windows.Forms.Label label22;
        private System.Windows.Forms.GroupBox RobotFeedbackBox;
        internal System.Windows.Forms.Label Temp4;
        internal System.Windows.Forms.Label Volt4;
        internal System.Windows.Forms.Label Load4;
        internal System.Windows.Forms.Label Vel4;
        internal System.Windows.Forms.Label Pos4;
        internal System.Windows.Forms.Label arm_label;
        internal System.Windows.Forms.Label label110;
        internal System.Windows.Forms.Label Temp5;
        internal System.Windows.Forms.Label Volt5;
        internal System.Windows.Forms.Label Load5;
        internal System.Windows.Forms.Label Vel5;
        internal System.Windows.Forms.Label Pos5;
        internal System.Windows.Forms.Label Temp3;
        internal System.Windows.Forms.Label Volt3;
        internal System.Windows.Forms.Label Load3;
        internal System.Windows.Forms.Label Vel3;
        internal System.Windows.Forms.Label Pos3;
        internal System.Windows.Forms.Label Temp2;
        internal System.Windows.Forms.Label Volt2;
        internal System.Windows.Forms.Label Load2;
        internal System.Windows.Forms.Label Vel2;
        internal System.Windows.Forms.Label Pos2;
        internal System.Windows.Forms.Label Temp1;
        internal System.Windows.Forms.Label Volt1;
        internal System.Windows.Forms.Label Load1;
        internal System.Windows.Forms.Label Vel1;
        internal System.Windows.Forms.Label Pos1;
        internal System.Windows.Forms.Label label109;
        internal System.Windows.Forms.Label label108;
        internal System.Windows.Forms.Label label107;
        internal System.Windows.Forms.Label label106;
        internal System.Windows.Forms.Label label200;
        internal System.Windows.Forms.GroupBox LEDbox;
        internal System.Windows.Forms.Button LEDdisconnect;
        internal System.Windows.Forms.Button LEDconnect;
        internal System.Windows.Forms.GroupBox SimBox;
        internal System.Windows.Forms.Button SIMdcBTN;
        internal System.Windows.Forms.Button SIMconnectBTN;
        internal System.Windows.Forms.Button openSim;
        internal System.Windows.Forms.NumericUpDown cctime_ctrl;
        internal System.Windows.Forms.Label label2;
        internal System.Windows.Forms.Timer Timer1;
        internal System.Windows.Forms.Timer Timer3;
        internal System.Windows.Forms.SaveFileDialog SaveFileDialog1;
        internal System.Windows.Forms.OpenFileDialog OpenFileDialog1;
        internal System.Windows.Forms.HelpProvider HelpProvider1;
        internal System.Windows.Forms.Timer Timer2;
        private System.Windows.Forms.CheckBox sim_flag;
        internal System.Windows.Forms.Label RAM_text;
        internal System.Windows.Forms.Label label29;
        internal System.Windows.Forms.Label label35;
        internal System.Windows.Forms.ComboBox switch5_dofmode_box;
        internal System.Windows.Forms.Label label34;
        internal System.Windows.Forms.ComboBox switch4_dofmode_box;
        internal System.Windows.Forms.Label label32;
        internal System.Windows.Forms.ComboBox switch3_dofmode_box;
        internal System.Windows.Forms.Label label30;
        internal System.Windows.Forms.ComboBox switch2_dofmode_box;
        internal System.Windows.Forms.Label label24;
        internal System.Windows.Forms.ComboBox switch1_dofmode_box;
        internal System.Windows.Forms.GroupBox groupBox2;
        internal System.Windows.Forms.Button button4;
        internal System.Windows.Forms.Button button5;
        internal System.Windows.Forms.Button button6;
        internal System.Windows.Forms.Label label36;
        private System.Windows.Forms.ComboBox cmbSerialPorts;
        private System.Windows.Forms.GroupBox MLBox;
        internal System.Windows.Forms.Button home_BTN;
        internal System.Windows.Forms.Button torque_off;
        internal System.Windows.Forms.Button torque_on;
        internal System.Windows.Forms.Button MLdisable;
        internal System.Windows.Forms.Button MLenable;
        private System.Windows.Forms.Button ML_start;
        internal System.Windows.Forms.NumericUpDown hand_w;
        internal System.Windows.Forms.NumericUpDown hand_p;
        internal System.Windows.Forms.Label label40;
        internal System.Windows.Forms.NumericUpDown wristRot_w;
        internal System.Windows.Forms.Label label42;
        internal System.Windows.Forms.NumericUpDown wristRot_p;
        internal System.Windows.Forms.NumericUpDown elbow_w;
        internal System.Windows.Forms.NumericUpDown elbow_p;
        internal System.Windows.Forms.Label label53;
        internal System.Windows.Forms.NumericUpDown shoulder_w;
        internal System.Windows.Forms.NumericUpDown shoulder_p;
        internal System.Windows.Forms.Label label54;
        internal System.Windows.Forms.Label label55;
        internal System.Windows.Forms.NumericUpDown wristFlex_w;
        internal System.Windows.Forms.NumericUpDown wristFlex_p;
        internal System.Windows.Forms.Label label58;
        internal System.Windows.Forms.Label label60;
        internal System.Windows.Forms.Label label114;
        internal System.Windows.Forms.Label label115;
        internal System.Windows.Forms.Label label105;
        internal System.Windows.Forms.Label label111;
        internal System.Windows.Forms.Label label112;
        internal System.Windows.Forms.Label label113;
        private System.Windows.Forms.CheckBox checkShoulderLeft;
        internal System.Windows.Forms.Label label86;
        internal System.Windows.Forms.Label label87;
        internal System.Windows.Forms.Label label88;
        internal System.Windows.Forms.Label label85;
        internal System.Windows.Forms.Label label84;
        internal System.Windows.Forms.Label label82;
        internal System.Windows.Forms.Label label81;
        internal System.Windows.Forms.Label label80;
        internal System.Windows.Forms.Label label76;
        internal System.Windows.Forms.Label label73;
        internal System.Windows.Forms.Label label71;
        internal System.Windows.Forms.Label label69;
        internal System.Windows.Forms.Label label62;
        internal System.Windows.Forms.Label label61;
        internal System.Windows.Forms.Label label59;
        private System.Windows.Forms.CheckBox checkGuide;
        private System.Windows.Forms.Label labelStickRightY;
        private System.Windows.Forms.Label labelStickRightX;
        private System.Windows.Forms.Label labelStickLeftY;
        private System.Windows.Forms.Label labelStickLeftX;
        private System.Windows.Forms.Label labelTriggerRight;
        private System.Windows.Forms.Label labelTriggerLeft;
        private System.Windows.Forms.CheckBox checkDPadLeft;
        private System.Windows.Forms.CheckBox checkDPadDown;
        private System.Windows.Forms.CheckBox checkDPadRight;
        private System.Windows.Forms.CheckBox checkDPadUp;
        private System.Windows.Forms.CheckBox checkStickLeft;
        private System.Windows.Forms.CheckBox checkStickRight;
        private System.Windows.Forms.CheckBox checkBack;
        private System.Windows.Forms.CheckBox checkStart;
        private System.Windows.Forms.CheckBox checkA;
        private System.Windows.Forms.CheckBox checkB;
        private System.Windows.Forms.CheckBox checkX;
        private System.Windows.Forms.CheckBox checkY;
        private System.Windows.Forms.CheckBox checkShoulderRight;
        private System.ComponentModel.BackgroundWorker pollingWorker;
        private System.Windows.Forms.Button dynaDisconnect;
        private System.Windows.Forms.Button dynaConnect;
        private System.Windows.Forms.Button LEDoff;
        private System.Windows.Forms.Button LEDon;
        private System.Windows.Forms.Button TorqueOff;
        private System.Windows.Forms.Button TorqueOn;
        private System.Windows.Forms.Button moveCCW;
        private System.Windows.Forms.Button moveCW;
        internal System.Windows.Forms.Label label116;
        private System.Windows.Forms.ComboBox comboBox1;
        internal System.Windows.Forms.Label delay;
        internal System.Windows.Forms.Label label117;
        internal System.Windows.Forms.Label label118;
        internal System.Windows.Forms.Label label120;
        internal System.Windows.Forms.Label dynaError;
        internal System.Windows.Forms.Label dynaCommResult;
        private System.Windows.Forms.Button readFeedback;
        internal System.Windows.Forms.Label delay_max;
        internal System.Windows.Forms.Label label121;
        private System.Windows.Forms.Label dynaStatus;
        private System.Windows.Forms.Label label119;
        private System.Windows.Forms.Button cmbSerialRefresh;
        private System.Windows.Forms.GroupBox BentoGroupBox;
        private System.Windows.Forms.GroupBox xBoxGroupBox;
        private System.Windows.Forms.Button XboxDisconnect;
        private System.Windows.Forms.Button XboxConnect;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.GroupBox groupBox6;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.ToolStripMenuItem mappingGraphicToolStripMenuItem;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.ComboBox comboBox2;
        private System.Windows.Forms.CheckedListBox checkedListFruit;
        private System.Windows.Forms.GroupBox MYOgroupBox;
        private System.Windows.Forms.Label myo_ch1;
        private System.Windows.Forms.Label myo_ch2;
        internal System.Windows.Forms.Label label134;
        internal System.Windows.Forms.Label label136;
        private System.Windows.Forms.Label myo_ch3;
        private System.Windows.Forms.Label myo_ch4;
        private System.Windows.Forms.Label myo_ch5;
        private System.Windows.Forms.Label myo_ch6;
        private System.Windows.Forms.Label myo_ch7;
        internal System.Windows.Forms.Label label128;
        private System.Windows.Forms.Label myo_ch8;
        internal System.Windows.Forms.Label label130;
        internal System.Windows.Forms.Label label131;
        internal System.Windows.Forms.Label label133;
        internal System.Windows.Forms.Label label135;
        internal System.Windows.Forms.Label label137;
        private System.Windows.Forms.GroupBox groupBox8;
        private System.Windows.Forms.Button KBconnect;
        private System.Windows.Forms.Button KBdisconnect;
        private System.Windows.Forms.GroupBox groupBox7;
        private System.Windows.Forms.Button MYOconnect;
        private System.Windows.Forms.Button MYOdisconnect;
        private System.Windows.Forms.GroupBox KBgroupBox;
        private System.Windows.Forms.CheckBox KBcheckD;
        private System.Windows.Forms.CheckBox KBcheckS;
        private System.Windows.Forms.CheckBox KBcheckA;
        private System.Windows.Forms.CheckBox KBcheckW;
        internal System.Windows.Forms.Label label122;
        internal System.Windows.Forms.Label label123;
        internal System.Windows.Forms.Label label124;
        internal System.Windows.Forms.Label label125;
        private System.Windows.Forms.CheckBox KBcheckRightAlt;
        private System.Windows.Forms.CheckBox KBcheckSpace;
        private System.Windows.Forms.CheckBox KBcheckLeftAlt;
        internal System.Windows.Forms.Label label142;
        internal System.Windows.Forms.Label label143;
        internal System.Windows.Forms.Label label144;
        private System.Windows.Forms.CheckBox KBcheckRight;
        private System.Windows.Forms.CheckBox KBcheckDown;
        private System.Windows.Forms.CheckBox KBcheckLeft;
        private System.Windows.Forms.CheckBox KBcheckUp;
        internal System.Windows.Forms.Label label138;
        internal System.Windows.Forms.Label label139;
        internal System.Windows.Forms.Label label140;
        internal System.Windows.Forms.Label label141;
        private System.Windows.Forms.CheckBox KBcheckSemiColon;
        private System.Windows.Forms.CheckBox KBcheckL;
        private System.Windows.Forms.CheckBox KBcheckK;
        private System.Windows.Forms.CheckBox KBcheckO;
        internal System.Windows.Forms.Label label126;
        internal System.Windows.Forms.Label label127;
        internal System.Windows.Forms.Label label129;
        internal System.Windows.Forms.Label label132;
        private System.Windows.Forms.CheckBox KBcheckRamp;
        internal System.Windows.Forms.Label KBlabelRamp;
        private System.Windows.Forms.Label KBrampS;
        private System.Windows.Forms.Label KBrampD;
        private System.Windows.Forms.Label KBrampW;
        private System.Windows.Forms.Label KBrampA;
        private System.Windows.Forms.Button ML_stop;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabIO;
        private System.Windows.Forms.TabPage tabMapping;
        private System.Windows.Forms.TabPage tabBento;
        private System.Windows.Forms.TabPage tabXPC;
        private System.Windows.Forms.TabPage tabViz;
        private System.Windows.Forms.CheckedListBox XBoxList;
        private System.Windows.Forms.Button XBoxClearAll;
        private System.Windows.Forms.Button XBoxSelectAll;
        private System.Windows.Forms.GroupBox groupBox9;
        private System.Windows.Forms.PictureBox pictureBox10;
        private System.Windows.Forms.Button MYOclearAll;
        private System.Windows.Forms.Button MYOselectAll;
        private System.Windows.Forms.CheckedListBox MYOlist;
        private System.Windows.Forms.PictureBox pictureBox11;
        private System.Windows.Forms.Button KBclearAll;
        private System.Windows.Forms.CheckedListBox KBlist;
        private System.Windows.Forms.Button KBselectAll;
        private System.Windows.Forms.PictureBox pictureBox9;
        private System.Windows.Forms.PictureBox pictureBox12;
        private System.Windows.Forms.Button BentoClearAll;
        private System.Windows.Forms.Button BentoSelectAll;
        private System.Windows.Forms.CheckedListBox BentoList;
        internal System.Windows.Forms.Label label156;
        internal System.Windows.Forms.Label label157;
        internal System.Windows.Forms.Label label158;
        internal System.Windows.Forms.Label label163;
        internal System.Windows.Forms.Label label162;
        internal System.Windows.Forms.Label label150;
        internal System.Windows.Forms.Label label146;
        internal System.Windows.Forms.GroupBox groupBox16;
        internal System.Windows.Forms.Label label237;
        internal System.Windows.Forms.ComboBox switch5MappingBox;
        internal System.Windows.Forms.Label label238;
        internal System.Windows.Forms.ComboBox switch4MappingBox;
        internal System.Windows.Forms.Label label239;
        internal System.Windows.Forms.ComboBox switch3MappingBox;
        internal System.Windows.Forms.Label label240;
        internal System.Windows.Forms.ComboBox switch2MappingBox;
        internal System.Windows.Forms.Label label241;
        internal System.Windows.Forms.NumericUpDown switchTimeCtrl1;
        internal System.Windows.Forms.ComboBox switch1MappingBox;
        internal System.Windows.Forms.Label label242;
        private System.Windows.Forms.CheckBox vocalBox;
        private System.Windows.Forms.CheckBox dingBox;
        private System.Windows.Forms.CheckBox switch5Flip;
        private System.Windows.Forms.CheckBox switch4Flip;
        private System.Windows.Forms.CheckBox switch3Flip;
        private System.Windows.Forms.CheckBox switch2Flip;
        private System.Windows.Forms.CheckBox switch1Flip;
        internal System.Windows.Forms.Label label253;
        internal System.Windows.Forms.ComboBox switch5OutputBox;
        internal System.Windows.Forms.ComboBox switch4OutputBox;
        internal System.Windows.Forms.ComboBox switch3OutputBox;
        internal System.Windows.Forms.ComboBox switch2OutputBox;
        internal System.Windows.Forms.ComboBox switch1OutputBox;
        internal System.Windows.Forms.Label switchLabel;
        internal System.Windows.Forms.Label label257;
        internal System.Windows.Forms.ComboBox switchDoFbox;
        internal System.Windows.Forms.Label label258;
        private System.Windows.Forms.Label labelID;
        private System.Windows.Forms.Label labelText;
        internal System.Windows.Forms.Button button14;
        private System.Windows.Forms.CheckedListBox checkedListDairy;
        private System.Windows.Forms.Label labelType;
        internal System.Windows.Forms.ComboBox InputComboBox;
        internal System.Windows.Forms.ComboBox OutputComboBox;
        internal System.Windows.Forms.Label label166;
        private brachIOplexus.DoF doF1;
        private brachIOplexus.DoF doF6;
        private brachIOplexus.DoF doF5;
        private brachIOplexus.DoF doF4;
        private brachIOplexus.DoF doF3;
        private brachIOplexus.DoF doF2;
        private System.Windows.Forms.Button BentoRun;
        private System.Windows.Forms.Button BentoSuspend;
        internal System.Windows.Forms.Label label37;
        internal System.Windows.Forms.Label label38;
        internal System.Windows.Forms.ComboBox switchInputBox;
        internal System.Windows.Forms.Label label39;
        internal System.Windows.Forms.Label label27;
        internal System.Windows.Forms.ComboBox switchModeBox;
        private System.Windows.Forms.GroupBox groupBox11;
        private System.Windows.Forms.GroupBox groupBox10;
        internal System.Windows.Forms.Label label103;
        internal System.Windows.Forms.Label label104;
        internal System.Windows.Forms.Label label145;
        internal System.Windows.Forms.Label label147;
        private System.Windows.Forms.Label switchSmaxLabel1;
        private System.Windows.Forms.Label switchSminLabel1;
        private System.Windows.Forms.Label switchSminTick1;
        private System.Windows.Forms.Label switchSmaxTick1;
        public System.Windows.Forms.NumericUpDown switchSmaxCtrl1;
        public System.Windows.Forms.NumericUpDown switchSminCtrl1;
        public System.Windows.Forms.ProgressBar switchSignalBar1;
        public System.Windows.Forms.NumericUpDown switchGainCtrl1;
        private System.Windows.Forms.CheckBox myoBuzzBox;
        internal System.Windows.Forms.Label ID2_state;
        internal System.Windows.Forms.Label label148;
        private System.Windows.Forms.Panel statusPanel1;
        private System.Windows.Forms.Button BentoRunStatus;
        private System.Windows.Forms.Label BentoStatus;
        private System.Windows.Forms.Label label149;
        private System.Windows.Forms.Label BentoErrorText;
        private System.Windows.Forms.Label BentoErrorColor;
        private System.Windows.Forms.CheckBox BentoAdaptGripCheck;
        internal System.Windows.Forms.NumericUpDown BentoAdaptGripCtrl;
        private System.Windows.Forms.GroupBox BentoEnvLimitsBox;
        internal System.Windows.Forms.Label label155;
        internal System.Windows.Forms.NumericUpDown numericUpDown3;
        internal System.Windows.Forms.Label label153;
        private System.Windows.Forms.CheckBox environCheck;
        internal System.Windows.Forms.NumericUpDown numericUpDown2;
        internal System.Windows.Forms.Label label154;
        private System.Windows.Forms.GroupBox BentoAdaptGripBox;
        internal System.Windows.Forms.Label label152;
        internal System.Windows.Forms.Label label151;
        internal System.Windows.Forms.Label label159;
        internal System.Windows.Forms.Label label160;
        private System.Windows.Forms.GroupBox groupBox14;
        private System.Windows.Forms.CheckBox textBox;
        private System.Windows.Forms.GroupBox groupBox13;
        private System.Windows.Forms.CheckBox XboxBuzzBox;
        private System.Windows.Forms.GroupBox groupBox12;
        private System.Windows.Forms.Label MYOstatus;
        private System.Windows.Forms.Label label164;
        private System.Windows.Forms.ToolStripMenuItem xBoxToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem mYOSequentialLeftToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem mYOSequentialRightToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem keyboardMultijointToolStripMenuItem;
        private System.Windows.Forms.GroupBox groupBox15;
        private System.Windows.Forms.Button biopatrecClearAll;
        private System.Windows.Forms.Button biopatrecConnect;
        private System.Windows.Forms.Button biopatrecSelectAll;
        private System.Windows.Forms.Button biopatrecDisconnect;
        private System.Windows.Forms.CheckedListBox biopatrecList;
        private System.Windows.Forms.GroupBox groupBox17;
        private System.Windows.Forms.Button SLRTclearAll;
        private System.Windows.Forms.Button SLRTconnect;
        private System.Windows.Forms.Button SLRTselectAll;
        private System.Windows.Forms.Button SLRTdisconnect;
        private System.Windows.Forms.CheckedListBox SLRTlist;
        private System.Windows.Forms.GroupBox SLRTgroupBox;
        private System.Windows.Forms.Label slrt_ch1;
        private System.Windows.Forms.Label slrt_ch2;
        internal System.Windows.Forms.Label label167;
        internal System.Windows.Forms.Label label168;
        private System.Windows.Forms.Label slrt_ch3;
        private System.Windows.Forms.Label slrt_ch4;
        private System.Windows.Forms.Label slrt_ch5;
        private System.Windows.Forms.Label slrt_ch6;
        private System.Windows.Forms.Label slrt_ch7;
        internal System.Windows.Forms.Label label174;
        private System.Windows.Forms.Label slrt_ch8;
        internal System.Windows.Forms.Label label176;
        internal System.Windows.Forms.Label label177;
        internal System.Windows.Forms.Label label178;
        internal System.Windows.Forms.Label label179;
        internal System.Windows.Forms.Label label180;
        private System.Windows.Forms.GroupBox biopatrecGroupBox;
        internal System.Windows.Forms.Label label184;
        internal System.Windows.Forms.Label label182;
        private System.Windows.Forms.CheckBox BPRclass12;
        internal System.Windows.Forms.Label label165;
        private System.Windows.Forms.CheckBox BPRclass24;
        internal System.Windows.Forms.Label label169;
        private System.Windows.Forms.CheckBox BPRclass23;
        private System.Windows.Forms.CheckBox BPRclass17;
        private System.Windows.Forms.CheckBox BPRclass18;
        private System.Windows.Forms.CheckBox BPRclass21;
        private System.Windows.Forms.CheckBox BPRclass20;
        private System.Windows.Forms.CheckBox BPRclass19;
        private System.Windows.Forms.CheckBox BPRclass22;
        internal System.Windows.Forms.Label label170;
        internal System.Windows.Forms.Label label171;
        internal System.Windows.Forms.Label label172;
        internal System.Windows.Forms.Label label173;
        internal System.Windows.Forms.Label label175;
        internal System.Windows.Forms.Label label181;
        internal System.Windows.Forms.Label label201;
        private System.Windows.Forms.CheckBox BPRclass11;
        internal System.Windows.Forms.Label label161;
        private System.Windows.Forms.CheckBox BPRclass10;
        private System.Windows.Forms.CheckBox BPRclass3;
        private System.Windows.Forms.CheckBox BPRclass2;
        private System.Windows.Forms.CheckBox BPRclass1;
        private System.Windows.Forms.CheckBox BPRclass0;
        private System.Windows.Forms.CheckBox BPRclass4;
        private System.Windows.Forms.CheckBox BPRclass5;
        private System.Windows.Forms.CheckBox BPRclass8;
        private System.Windows.Forms.CheckBox BPRclass7;
        private System.Windows.Forms.CheckBox BPRclass13;
        private System.Windows.Forms.CheckBox BPRclass14;
        private System.Windows.Forms.CheckBox BPRclass15;
        private System.Windows.Forms.CheckBox BPRclass16;
        private System.Windows.Forms.CheckBox BPRclass6;
        internal System.Windows.Forms.Label label183;
        internal System.Windows.Forms.Label label185;
        internal System.Windows.Forms.Label label187;
        internal System.Windows.Forms.Label label189;
        private System.Windows.Forms.CheckBox BPRclass9;
        internal System.Windows.Forms.Label label190;
        internal System.Windows.Forms.Label label191;
        internal System.Windows.Forms.Label label192;
        internal System.Windows.Forms.Label label193;
        internal System.Windows.Forms.Label label194;
        internal System.Windows.Forms.Label label195;
        internal System.Windows.Forms.Label label196;
        internal System.Windows.Forms.Label label197;
        internal System.Windows.Forms.Label label198;
        internal System.Windows.Forms.Label label199;
        internal System.Windows.Forms.ComboBox biopatrecMode;
        internal System.Windows.Forms.Label label202;
        internal System.Windows.Forms.TextBox biopatrecIPport;
        internal System.Windows.Forms.Label label186;
        internal System.Windows.Forms.Label label188;
        internal System.Windows.Forms.TextBox biopatrecIPaddr;
        internal System.Windows.Forms.Label label203;
        internal System.Windows.Forms.Label biopatrecDelay;
        private System.Windows.Forms.Button demoShutdownButton;
        private System.Windows.Forms.Button demoMYObutton;
        private System.Windows.Forms.Button demoXBoxButton;
        private System.IO.Ports.SerialPort serialArduinoInput;
        private System.Windows.Forms.GroupBox groupBox18;
        private System.Windows.Forms.ComboBox ArduinoInputCOM;
        internal System.Windows.Forms.Label label204;
        private System.Windows.Forms.Button ArduinoInputClearAll;
        private System.Windows.Forms.Button ArduinoInputConnect;
        private System.Windows.Forms.Button ArduinoInputSelectAll;
        private System.Windows.Forms.Button ArduinoInputDisconnect;
        private System.Windows.Forms.CheckedListBox ArduinoInputList;
        private System.Windows.Forms.GroupBox ArduinoInputGroupBox;
        private System.Windows.Forms.Label arduino_A0;
        private System.Windows.Forms.Label arduino_A1;
        internal System.Windows.Forms.Label label207;
        internal System.Windows.Forms.Label label208;
        private System.Windows.Forms.Label arduino_A2;
        private System.Windows.Forms.Label arduino_A3;
        private System.Windows.Forms.Label arduino_A4;
        private System.Windows.Forms.Label arduino_A5;
        private System.Windows.Forms.Label arduino_A6;
        internal System.Windows.Forms.Label label214;
        private System.Windows.Forms.Label arduino_A7;
        internal System.Windows.Forms.Label label216;
        internal System.Windows.Forms.Label label217;
        internal System.Windows.Forms.Label label218;
        internal System.Windows.Forms.Label label219;
        internal System.Windows.Forms.Label label220;
        private System.Windows.Forms.Label switchSmaxLabel2;
        private System.Windows.Forms.Label switchSminLabel2;
        private System.Windows.Forms.Label switchSminTick2;
        private System.Windows.Forms.Label switchSmaxTick2;
        public System.Windows.Forms.NumericUpDown switchSmaxCtrl2;
        public System.Windows.Forms.NumericUpDown switchSminCtrl2;
        public System.Windows.Forms.ProgressBar switchSignalBar2;
        public System.Windows.Forms.NumericUpDown switchGainCtrl2;
        internal System.Windows.Forms.NumericUpDown switchTimeCtrl2;
        internal System.Windows.Forms.Label label205;
        internal System.Windows.Forms.Label switchState_label;
        internal System.Windows.Forms.Label label213;
        internal System.Windows.Forms.Label flag2_label;
        internal System.Windows.Forms.Label label211;
        internal System.Windows.Forms.Label flag1_label;
        internal System.Windows.Forms.Label label209;
        internal System.Windows.Forms.Label timer1_label;
        private System.Windows.Forms.GroupBox groupBox19;
        internal System.Windows.Forms.Button BentoProfileOpen;
        internal System.Windows.Forms.ComboBox BentoProfileBox;
        internal System.Windows.Forms.Button BentoProfileSave;
        private System.Windows.Forms.GroupBox AutoLevellingBox;
        private System.Windows.Forms.GroupBox FlexionPIDBox;
        internal System.Windows.Forms.Label CurrentFlexion;
        internal System.Windows.Forms.Label label224;
        internal System.Windows.Forms.Label label223;
        internal System.Windows.Forms.Label SetpointFlexion;
        internal System.Windows.Forms.NumericUpDown Kd_theta_ctrl;
        internal System.Windows.Forms.Label label210;
        internal System.Windows.Forms.NumericUpDown Ki_theta_ctrl;
        internal System.Windows.Forms.Label label206;
        internal System.Windows.Forms.NumericUpDown Kp_theta_ctrl;
        internal System.Windows.Forms.Label label212;
        private System.Windows.Forms.GroupBox RotationPIDBox;
        internal System.Windows.Forms.Label CurrentRotation;
        internal System.Windows.Forms.Label label222;
        internal System.Windows.Forms.Label label221;
        internal System.Windows.Forms.NumericUpDown Kd_phi_ctrl;
        internal System.Windows.Forms.Label SetpointRotation;
        internal System.Windows.Forms.Label label215;
        internal System.Windows.Forms.NumericUpDown Ki_phi_ctrl;
        internal System.Windows.Forms.Label label225;
        internal System.Windows.Forms.NumericUpDown Kp_phi_ctrl;
        internal System.Windows.Forms.Label label226;
        private System.Windows.Forms.CheckBox AL_Enabled;
        private System.Windows.Forms.Label label228;
        private System.Windows.Forms.NumericUpDown log_number;
        private System.Windows.Forms.GroupBox LoggingGroupBox;
        private System.Windows.Forms.Button StartLogging;
        private System.Windows.Forms.Button StopLogging;
        private System.Windows.Forms.TextBox intervention;
        private System.Windows.Forms.Label label229;
        private System.Windows.Forms.TextBox task_type;
        private System.Windows.Forms.Label label227;
        private System.Windows.Forms.TextBox ppt_no;
        private System.Windows.Forms.Label label_ppt_no;
        internal System.Windows.Forms.Label label234;
        internal System.Windows.Forms.Label label233;
        internal System.Windows.Forms.Label label232;
        internal System.Windows.Forms.Label label231;
        internal System.Windows.Forms.Label label230;
        private System.Windows.Forms.CheckBox LogPID_Enabled;
        private System.Windows.Forms.CheckBox NN_PID_Enabled;
    }
}

