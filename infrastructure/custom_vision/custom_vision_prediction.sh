az login --tenant cf36141c-ddd7-45a7-b073-111f66d0b30c

az cognitiveservices account create \
    --name sbirkamlcvprediction \
    --resource-group sbirkamlrg \
    --kind CustomVision.Prediction \
    --sku S0 \
    --location westeurope \
    --yes