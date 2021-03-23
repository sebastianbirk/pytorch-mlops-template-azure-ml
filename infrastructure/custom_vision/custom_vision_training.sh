az login --tenant cf36141c-ddd7-45a7-b073-111f66d0b30c

az cognitiveservices account create \
    --name sbirkamlcvtraining \
    --resource-group sbirkamlrg \
    --kind CustomVision.Training \
    --sku S0 \
    --location westeurope \
    --yes