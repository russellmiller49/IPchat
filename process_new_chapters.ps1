# process_new_chapters.ps1
# Run textbook extraction only on chapters that have not yet been processed.
# Uses conda environment "ipchat"

# Activate conda environment
conda activate ipchat

# Loop through all PDFs in Textbooks\Chapter pdfs
Get-ChildItem "Textbooks\Chapter pdfs" -Filter *.pdf | ForEach-Object {
    $chapter = $_.BaseName
    $goldFile = "data\gold_standard_extractions\$chapter`_gold_standard.json"
    $jsonFile = "Textbooks\Chapter json\$chapter.json"

    if (-not (Test-Path $goldFile)) {
        Write-Host "▶ Processing $chapter..."
        if (Test-Path $jsonFile) {
            python .\tools\gold_standard_pipeline.py --single "$($_.FullName)" --adobe-json "$jsonFile" --model gpt-5 --verbose
        } else {
            python .\tools\gold_standard_pipeline.py --single "$($_.FullName)" --model gpt-5 --verbose
        }
    } else {
        Write-Host "✔ Skipping $chapter (already processed)"
    }
}
