using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CollisionStateManager : MonoBehaviour
{
    // private Car carComponent;
    RLAgent agent;
    Car_Controller carController;

    // public 
    public Transform rewardCoinTransform;
    public Transform agentTransform;

    public float minX;
    public float maxX;
    public float minY;
    public float maxY;
    public float minZ;
    public float maxZ;

    public int collisionType; // -1: 벽, 0: 충돌x, 1: 코인
    public bool isAllCoinColledted;
    public bool isDone;

    // private Vector3 initialRewardCoinPosition;
    public GameObject RewardCoins;

    private Vector3 initialRewardCoinsPosition;
    private Quaternion initialRewardCoinsRotation;


    private Vector3 initialAgentPosition;
    private Quaternion initialAgentRotation;

    
    void Start() 
    {
        // initialRewardCoinPosition = rewardCoinTransform.position;
        initialRewardCoinsPosition = rewardCoinTransform.position;
        initialRewardCoinsRotation = rewardCoinTransform.rotation;

        initialAgentPosition = agentTransform.position;
        initialAgentRotation = agentTransform.rotation;

        collisionType = 0;
        isDone = false;
        carController = GetComponent<Car_Controller>();
        agent = new RLAgent();

        isAllCoinColledted = false;

    }

    
    void Update() {
        // Debug.Log(agentTransform.position);
        // Debug.Log(rewardCoinTransform.position);
    }

    public void endEpisode() { 
        Debug.Log("end!");
        agentTransform.position = initialAgentPosition;
        agentTransform.rotation = initialAgentRotation;

        carController.Car_Rigidbody.velocity = Vector3.zero;
        carController.Current_Virtual_Key = "H";
        collisionType = 0;
        isDone = false;

        isAllCoinColledted = false;

        int childCount = RewardCoins.transform.childCount;

        for (int i = 0; i < childCount; i++) {
            Transform childTransform = RewardCoins.transform.GetChild(i);
            GameObject childObject = childTransform.gameObject;
            childObject.SetActive(true);
        }

    }

    
    
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Coin"))
        {
            int coinCnt = GameObject.FindGameObjectsWithTag("Coin").Length;
            Debug.Log(coinCnt);
            if (coinCnt == 0) {
                isAllCoinColledted = true;
                isDone = true;
            }
            collisionType = 1;
            // Destroy(collision.gameObject);
            collision.gameObject.SetActive(false);
            // endEpisode(); 
        }
        
        if (collision.gameObject.CompareTag("Wall"))
        {
            collisionType = -1;
            isDone = true;
            // endEpisode();
        }
    }
}
