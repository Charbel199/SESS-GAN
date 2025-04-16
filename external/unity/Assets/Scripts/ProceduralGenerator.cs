using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProceduralGenerator : MonoBehaviour
{
    public BiomePreset[] biomes;
    public GameObject tilePrefab;

    [Header("Dimensions")]
    public int width = 18;
    public int height = 10;
    public float scale = 1.0f;
    public Vector2 offset;

    [Header("Height Map")]
    public Wave[] heightWaves;
    public float[,] heightMap;
    [Header("Road Map")]
    public Wave[] roadWaves;
    private float[,] roadMap;
    [Header("Grass Map")]
    public Wave[] grassWaves;
    private float[,] grassMap;
    void GenerateMap()
    {
        // height map
        heightMap = NoiseGenerator.Generate(width, height, scale, heightWaves, offset);
        // moisture map
        roadMap = NoiseGenerator.Generate(width, height, scale, roadWaves, offset);
        // heat map
        grassMap = NoiseGenerator.Generate(width, height, scale, grassWaves, offset);
        for (int x = 0; x < width; ++x)
        {
            for (int y = 0; y < height; ++y)
            {
                GameObject tile = Instantiate(tilePrefab, new Vector3(x - width/2 + 0.5f, y - height/2 + 0.5f, 0), Quaternion.identity);
                tile.GetComponent<SpriteRenderer>().sprite = GetBiome(heightMap[x, y], roadMap[x, y], grassMap[x, y]).GetTileSprite();
            }
        }
    }
    BiomePreset biomeToReturn;
    BiomePreset GetBiome(float height, float moisture, float heat)
    {
        List<BiomeTempData> biomeTemp = new List<BiomeTempData>();
        foreach (BiomePreset biome in biomes)
        {
            if (biome.MatchCondition(height, moisture, heat))
            {
                biomeTemp.Add(new BiomeTempData(biome));
            }
        }
        float curVal = 0.0f;
        foreach (BiomeTempData biome in biomeTemp)
        {
            if (biomeToReturn == null)
            {
                biomeToReturn = biome.biome;
                curVal = biome.GetDiffValue(height, moisture, heat);
            }
            else
            {
                if (biome.GetDiffValue(height, moisture, heat) < curVal)
                {
                    biomeToReturn = biome.biome;
                    curVal = biome.GetDiffValue(height, moisture, heat);
                }
            }
        }
        if (biomeToReturn == null)
            biomeToReturn = biomes[0];
        return biomeToReturn;
    }
    // Start is called before the first frame update
    void Start()
    {
        GenerateMap();
    }

    // Update is called once per frame
    void Update()
    {

    }
}
public class BiomeTempData
{
    public BiomePreset biome;
    public BiomeTempData(BiomePreset preset)
    {
        biome = preset;
    }

    public float GetDiffValue(float height, float road, float grass)
    {
        return (height - biome.minHeight) + (road - biome.minRoad) + (grass - biome.minGrass);
    }
}
