using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.IO;
using System;

public class DQNwithCNN : MonoBehaviour
{
    // Define the IP address and port number for the Python server
    private string ipAddress = "127.0.0.1";
    private int portNumber = 5000;

    // Define the socket and buffer for sending data to the Python server
    private Socket socket;
    private byte[] buffer = new byte[1024];

    // Define the game state variables
    private int screenWidth = 84;
    private int screenHeight = 84;
    private int numChannels = 4;
    private Texture2D screenTexture;
    private Color[] screenPixels;
    private byte[] gameState;

    void Start()
    {
        // Initialize the socket and connect to the Python server
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socket.Connect(new IPEndPoint(IPAddress.Parse(ipAddress), portNumber));

        // Initialize the screen texture and pixel array
        screenTexture = new Texture2D(screenWidth, screenHeight, TextureFormat.RGB24, false);
        screenPixels = new Color[screenWidth * screenHeight];
    }

    void Update()
    {
        // Update the screen texture and pixel array
        screenTexture.ReadPixels(new Rect(0, 0, screenWidth, screenHeight), 0, 0);
        screenTexture.Apply();
        screenPixels = screenTexture.GetPixels();

        // Convert the screen pixels to grayscale and downsample to 84x84
        for (int i = 0; i < screenPixels.Length; i++)
        {
            float grayscaleValue = (screenPixels[i].r + screenPixels[i].g + screenPixels[i].b) / 3.0f;
            screenPixels[i] = new Color(grayscaleValue, grayscaleValue, grayscaleValue);
        }
        Color[] resizedPixels = Downsample(screenPixels, screenWidth, screenHeight, 84, 84);

        // Convert the color array to a byte array and send it to the Python server
        gameState = new byte[resizedPixels.Length];
        for (int i = 0; i < resizedPixels.Length; i++)
        {
            gameState[i] = (byte)(resizedPixels[i].r * 255);
        }
        SendGameState(gameState);
    }

    // Downsample the color array to a smaller size using bilinear interpolation
    Color[] Downsample(Color[] pixels, int width, int height, int newWidth, int newHeight)
    {
        Color[] resizedPixels = new Color[newWidth * newHeight];
        float xRatio = (float)width / (float)newWidth;
        float yRatio = (float)height / (float)newHeight;
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                int x1 = (int)(x * xRatio);
                int x2 = x1 + 1;
                int y1 = (int)(y * yRatio);
                int y2 = y1 + 1;
                float xRatio1 = (float)(x2 - x) * xRatio;
                float xRatio2 = (float)(x - x1) * xRatio;
                float yRatio1 = (float)(y2 - y) * yRatio;
                float yRatio2 = (float)(y - y1) * yRatio;
                Color c1 = pixels[y1 * width + x1];
                Color c2 = pixels[y1 * width + x2];
                Color c3 = pixels[y2 * width + x1];
                Color c4 = pixels[y2 * width + x2];
                float r = c1.r * xRatio1 * yRatio1 + c2.r * xRatio2 * yRatio1 + c3.r * xRatio1 * yRatio2 + c4.r * xRatio2 * yRatio2;
                float g = c1.g * xRatio1 * yRatio1 + c2.g * xRatio2 * yRatio1 + c3.g * xRatio1 * yRatio2 + c4.g * xRatio2 * yRatio2;
                float b = c1.b * xRatio1 * yRatio1 + c2.b * xRatio2 * yRatio1 + c3.b * xRatio1 * yRatio2 + c4.b * xRatio2 * yRatio2;
                resizedPixels[y * newWidth + x] = new Color(r, g, b);
            }
        }
        return resizedPixels;
    }

    // Send the game state to the Python server using a socket connection
    void SendGameState(byte[] gameState)
    {
        int bytesSent = socket.Send(gameState);
    }

    void OnDestroy()
    {
        // Close the socket connection when the game object is destroyed
        socket.Close();
    }
}