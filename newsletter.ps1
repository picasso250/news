param(
    [switch]$dryrun,
    [string]$to = "wxiaochi@qq.com"
)

# 默认 dryrun=true，显式传 -dryrun:$false 才进入发送模式
$isDryrun = if ($PSBoundParameters.ContainsKey('dryrun')) { $dryrun } else { $true }

$draft  = "$env:TEMP\newsletter-draft.md"
$review = "$env:TEMP\newsletter-review.md"
$final  = "$env:TEMP\newsletter-final.md"

$prompt = (Get-Content -Raw "$PSScriptRoot\newsletter-prompt.md")
$prompt = $prompt.Replace('{DRAFT_PATH}',  $draft)
$prompt = $prompt.Replace('{REVIEW_PATH}', $review)
$prompt = $prompt.Replace('{FINAL_PATH}',  $final)
$prompt = $prompt.Replace('{TO_EMAIL}',    $to)

pi --print --mode json --model deepseek/deepseek-v4-flash $prompt

if (!$isDryrun) {
    python "$env:USERPROFILE\.agents\skills\send-email\scripts\send_email.py" `
        --to $to `
        --subject "AI 资讯简报 $(Get-Date -Format 'yyyy-MM-dd') — 反直觉 3 点" `
        --markdown-body-file $final
}
