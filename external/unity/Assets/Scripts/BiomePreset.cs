using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;


[CreateAssetMenu(fileName = "Biome Preset", menuName = "New Biome Preset")]
public class BiomePreset : ScriptableObject
{
    // Start is called before the first frame update
    public Sprite[] tiles;
    public float minHeight;
    public float minRoad;
    public float minGrass;

    public Sprite GetTileSprite()
    {
        return tiles[Random.Range(0, tiles.Length)];
    }
    void Start()
    {

    }

    public bool MatchCondition(float height, float road, float grass)
    {
        return height >= minHeight && road >= minRoad && grass >= minGrass;
    }

    // Update is called once per frame
    void Update()
    {

    }
}
