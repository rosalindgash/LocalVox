#define MyAppName "LocalVox"
#define MyAppVersion "0.1.0-dev"
#define MyAppPublisher "LocalVox contributors"
#define MyAppExeName "LocalVox.exe"

[Setup]
AppId={{5C6B56D5-8B4B-4D4A-B76E-4C7C7A54E8D8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\installer-output
OutputBaseFilename=LocalVox-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\LocalVox\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\LocalVox"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\LocalVox"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch LocalVox"; Flags: nowait postinstall skipifsilent
