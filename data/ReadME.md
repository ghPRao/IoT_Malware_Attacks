#### Download Data
<p align="center">
  <img width="1080" height="780" src="../images/ReadMe_UCI_Download_data.png">
</p>

<a href=https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT>Download from UCI Machine Learning Repository</a>

#### Data Directory Organization
The file structure and the content of the files are below:

```
├── N_BaIoT_dataset_description_v1.txt--------> Description about source of the data, information on features etc.
├── Ecobee_Thermostat-------------------------> IoT Device
│   ├── gafgyt_attacks------------------------> gafgyt attacks traffic types
│   │   ├── scan.csv-----> Scanning the network for vulnerable devices
│   │   ├── tcp.csv------> TCP flooding
│   │   ├── udp.csv------> UDP flooding
│   │   ├── junk.csv-----> Sending spam data
│   │   └── combo.csv---->Sending spam data and opening a connection to a specified IP address and port
│   ├── mirai_attacks----> Mirai attack traffic file
│   │   ├── syn.csv------> SYN flooding
│   │   ├── scan.csv-----> Scanning the network for vulnerable devices
│   │   ├── udpplain.csv-> UDPPLAIN flooding
│   │   ├── ack.csv------> ACK flooding
│   │   └── udp.csv------> UDP flooding
│   └── benign_traffic.csv--> Benign traffic observation for Ecobee Thermostat
├── SimpleHome_XCS7_1002_WHT_Security_Camera
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for SimpleHome XCS7-1002WHT Security Camera
├── Damini_Doorbell
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for Damini Doorbell
├── SimpleHome_XCS7_1003_WHT_Security_Camera
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for SimpleHome XCS7-1003WHT Security Camera
├── Samsung_SNH_1011_N_Webcam
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   └── benign_traffic.csv--> Benign traffic observation for  Samsung SNH-1011N Webcam
├── Ennino_Doorbell
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for Ennino_Doorbell
├── Philips_B120N10_Baby_Monitor
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for Philips B120N10 Baby Monitor
├── demonstrate_structure.csv --------> For demostration, all feature names are in this file
├── Provision_PT_737E_Security_Camera
│   ├── gafgyt_attacks
│   │   ├── scan.csv
│   │   ├── tcp.csv
│   │   ├── udp.csv
│   │   ├── junk.csv
│   │   └── combo.csv
│   ├── mirai_attacks
│   │   ├── syn.csv
│   │   ├── scan.csv
│   │   ├── udpplain.csv
│   │   ├── ack.csv
│   │   └── udp.csv
│   └── benign_traffic.csv--> Benign traffic observation for Provision PT737E Security Camera
└── Provision_PT_838_Security_Camera
    ├── gafgyt_attacks
    │   ├── scan.csv
    │   ├── tcp.csv
    │   ├── udp.csv
    │   ├── junk.csv
    │   └── combo.csv
    ├── mirai_attacks
    │   ├── syn.csv
    │   ├── scan.csv
    │   ├── udpplain.csv
    │   ├── ack.csv
    │   └── udp.csv
    └── benign_traffic.csv--> Benign traffic observation for Provision PT838 Security Camera
```
