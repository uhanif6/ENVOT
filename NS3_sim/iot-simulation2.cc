#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/energy-module.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("IoTSimulation");

void RunPythonScript(std::string cmd, double &totalLatency, double &totalEnergy) {
    std::array<char, 128> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }

    pclose(pipe);

    std::stringstream ss(result);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.find("Latency") != std::string::npos) {
            double latency = std::stod(line.substr(line.find(":") + 1));
            totalLatency += latency;
        }
        if (line.find("Energy Consumption") != std::string::npos) {
            double energy = std::stod(line.substr(line.find(":") + 1));
            totalEnergy += energy;
        }
    }
}

int main(int argc, char *argv[]) {
    CommandLine cmd;
    cmd.Parse(argc, argv);

    NS_LOG_INFO("Creating 250 IoT Nodes");

    NodeContainer wifiNodes;
    wifiNodes.Create(250);

    NodeContainer verifierNode;
    verifierNode.Create(1);

    NS_LOG_INFO("Setting up WiFi Network");

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(channel.Create());

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer wifiDevices;
    wifiDevices = wifi.Install(wifiPhy, wifiMac, wifiNodes);

    InternetStackHelper stack;
    stack.Install(wifiNodes);
    stack.Install(verifierNode);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer wifiInterfaces;
    wifiInterfaces = address.Assign(wifiDevices);

    NS_LOG_INFO("Setting Mobility Model");

    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(50.0, 50.0, 0.0)); // Verifier Node at center

    for (uint32_t i = 0; i < wifiNodes.GetN(); ++i) {
        positionAlloc->Add(Vector(150.0 * cos(2 * M_PI * i / wifiNodes.GetN()), 150.0 * sin(2 * M_PI * i / wifiNodes.GetN()), 0.0));
    }

    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiNodes);
    mobility.Install(verifierNode);

    NS_LOG_INFO("Setting Energy Model");

    BasicEnergySourceHelper energySourceHelper;
    energySourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(10000.0)); 
    ns3::energy::EnergySourceContainer sources = energySourceHelper.Install(wifiNodes);

    WifiRadioEnergyModelHelper radioEnergyHelper;
    radioEnergyHelper.Install(wifiDevices, sources);

    NS_LOG_INFO("Setting up Applications");

    double totalLatency = 0.0;
    double totalEnergy = 0.0;

    for (uint32_t i = 0; i < wifiNodes.GetN(); ++i) {
        Ptr<Node> node = wifiNodes.Get(i);
        std::string cmd = "python3 scratch/ml-model.py " + std::to_string(i);
        RunPythonScript(cmd, totalLatency, totalEnergy);
    }

    std::cout << "Total Latency: " << totalLatency << " ms" << std::endl;
    std::cout << "Total Energy Consumption: " << totalEnergy << " mJ" << std::endl;

    if (totalLatency < 1000 && totalEnergy < 0.5) {
        std::cout << "Constraints met: Latency and Energy Consumption are within limits." << std::endl;
    } else {
        std::cout << "Constraints not met: Latency or Energy Consumption exceeds limits." << std::endl;
    }

    Simulator::Stop(Seconds(10.0));

    NS_LOG_INFO("Running Simulation");
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete");

    return 0;
}
