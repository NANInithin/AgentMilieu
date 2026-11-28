Target amd64-unicode
!include "MUI2.nsh"
!include "x64.nsh"
!include "WinVer.nsh"

!ifndef APP_VERSION
  !define APP_VERSION "0.1.0"
!endif
!define UNINSTALL_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\AgentMilieu"

Name "AgentMilieu ${APP_VERSION}"
OutFile "../dist/windows/AgentMilieu-Setup.exe"
InstallDir "$LOCALAPPDATA\Programs\AgentMilieu"
RequestExecutionLevel user
SetCompressor /SOLID lzma
VIProductVersion "${APP_VERSION}.0"
VIAddVersionKey "ProductName" "AgentMilieu"
VIAddVersionKey "CompanyName" "AgentMilieu"
VIAddVersionKey "FileDescription" "AgentMilieu Windows Installer"
VIAddVersionKey "FileVersion" "${APP_VERSION}"
VIAddVersionKey "LegalCopyright" "AgentMilieu contributors"

!define MUI_FINISHPAGE_RUN "$INSTDIR\AgentMilieu.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Open AgentMilieu"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
    ${IfNot} ${RunningX64}
    ${OrIfNot} ${AtLeastWin10}
        MessageBox MB_OK|MB_ICONSTOP "AgentMilieu requires Windows 10 or 11 with x64 support."
        Abort
    ${EndIf}
    SetRegView 64
FunctionEnd

Section "AgentMilieu"
    SetShellVarContext current
    SetOutPath "$INSTDIR"
    File /r "../dist/AgentMilieu/*"
    File "../LICENSE"
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    CreateShortcut "$SMPROGRAMS\AgentMilieu.lnk" "$INSTDIR\AgentMilieu.exe"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayName" "AgentMilieu"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "DisplayIcon" "$INSTDIR\AgentMilieu.exe"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "Publisher" "AgentMilieu"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "URLInfoAbout" "https://agentmilieu.naninithin.me"
    WriteRegStr HKCU "${UNINSTALL_KEY}" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
    WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoModify" 1
    WriteRegDWORD HKCU "${UNINSTALL_KEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
    SetRegView 64
    SetShellVarContext current
    DeleteRegKey HKCU "${UNINSTALL_KEY}"
    Delete "$SMPROGRAMS\AgentMilieu.lnk"
    Delete "$INSTDIR\AgentMilieu.exe"
    RMDir /r "$INSTDIR\_internal"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"
SectionEnd
