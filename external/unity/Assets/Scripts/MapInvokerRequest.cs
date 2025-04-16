using UnityEngine;

[System.Serializable]
public class MapInvokerRequest
{
    public string matrixString;

    public static MapInvokerRequest CreateFromJSON(string jsonString)
    {
        return JsonUtility.FromJson<MapInvokerRequest>(jsonString);
    }

    // Given JSON input:
    // {"matrixString": "10230103"}
    // this example will return a MapInvokerRequest with matrixString="10230103
}