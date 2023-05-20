using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System;

public class SocketClient : MonoBehaviour
{
    public string ipAddress = "127.0.0.1";
    public int portNumber = 5000;

    // Define the socket and buffer for sending data to the Python server
    public Socket socket;
    public byte[] receiveBuffer = new byte[1024];


    private void Start() {
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        initSocket();
    }

    private void Update() {
        if (!isConnected()) {
            Debug.Log("try connect");
            // initSocket();
        }
        else {
          // Debug.Log("Connected!");
        }
    }

    public bool isConnected() {
        bool isConnected = socket != null && socket.Connected;
        return isConnected;
    }

    public void initSocket() 
    {
        
        socket.Connect(new IPEndPoint(IPAddress.Parse(ipAddress), portNumber));
    }

    public void SendStates(byte isDone, float reward, float[] currentPosition, byte[] Imagebytes)
    {
        // define combinedBytes
        byte[] combinedBytes = new byte[sizeof(byte) + sizeof(int) + currentPosition.Length * sizeof(float) + Imagebytes.Length];

        // make all parameter to bytes
        byte[] currentPositionBytes = new byte[currentPosition.Length * sizeof(float)];
        Buffer.BlockCopy(currentPosition, 0, currentPositionBytes, 0, currentPositionBytes.Length);

        // put all parameters into combinedBytes
        combinedBytes[0] = isDone;
        Buffer.BlockCopy(BitConverter.GetBytes(reward), 0, combinedBytes, sizeof(byte), sizeof(float));
        Buffer.BlockCopy(currentPositionBytes, 0, combinedBytes, sizeof(byte) + sizeof(float), currentPositionBytes.Length);
        Imagebytes.CopyTo(combinedBytes, sizeof(byte) + sizeof(float) + currentPositionBytes.Length);
        // Debug.Log(Imagebytes.Length);

        // Debug.Log(combinedBytes.Length);
        socket.Send(combinedBytes);
    }

    public string GetAction() 
    {
        int bytesReceived = socket.Receive(receiveBuffer);
        string response = Encoding.UTF8.GetString(receiveBuffer, 0, bytesReceived);
        return response;
    }

    void OnDestroy()
    {
        // Close the socket connection when the game object is destroyed
        socket.Close();
    }
}
