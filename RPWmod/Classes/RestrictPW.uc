//2017.4.16 ����J�n
//4.17 �֎~�p�[�N�̔���A�E�F�[�u���̔���͂ł���(�E�ցE)
//4.18 �֎~���킪����̂ŃJ�j������������ �� �L���X�g�[�b�I�֎~������J�j�������ł����I
//4.19 config�F�c�Ƃ肠�����`�ɂ͂Ȃ��������B
//4.20 ����Ɋւ��Ă͑��E�łȂ�������Ƃ����`�ɕύX�BDOSH�ɂ�镪��,���b�Z�[�W��ǉ��B
//4.22 �Ȃ��璍�����E�����Ă���E�E�E
//4.25 ��������Ɏl�ꔪ��A�悤�₭�`�ɂȂ邩�c�H
//4.27 ���O�̌����Ȃ񂶂���('A`) ����mod�̂����Ȃ̂������������A�y�������͂��Ă�����
//4.28 ���ς�炸�I�̃��O�͂Ƃꂸ �Ƃ肠�����A�[�}�[�ƃO���̕�[�@�\�����Ă݂�
//5.16 ����J�n����͂�ꃖ���c�����������ƂčX�V�ł� traderdash���Ă݂�
//6.03 �g���[�_�[���̃p�[�N�ύX������͖����ł���gg
//7.20 nam����̗v�]�ŊJ�n����dosh�ύX��ǉ����Ă݂�
//9.15 ���Ȃ�v�X�̍X�V nam����̗v�]������fakeplayers��ǉ����Ă݂�
//9.16 WaveSizeFakes�ɖ��O��ύX
//10.23 �����ɂ��maxmonsters�̕ύX�̐�����󂯂�
//10.25 2bosses��3bosses�ɂ��Ă݂�
//10.29 pro�̗v�]�ł��̏�Ńg���[�_�[�J����悤�ɂȂ���
//11.03 chatcommand�����I��������I
//11.10 �R���\�[�����b�Z�[�W���T�[�o�[�ł͓����Ȃ����� �ǂ������{��\������Ȃ��������� :o
//2018.2.11	2Boss�̐��`�A�����͂Ȃ炸�@�\���e�q��2�}�K�W����؂����玩���w���A�G���B�X�N�͓��ʂȏ���
//02.12	SetTimer(1.0, true, nameof(HackBroadcastHandler));�������Ă��c�c�^(^o^)�_��ú���

class RestrictPW extends KFMutator
	config(RestrictPW);

//<<---config�p�ϐ��̒�`--->>//

	/* Init Config */
		//config�t�@�C�����݂̊m�F
		var config bool bIWontInitThisConfig;
	/* Perk Settings */
		var config string MinPerkLevel_Berserker;
		var config string MinPerkLevel_Commando;
		var config string MinPerkLevel_Support;
		var config string MinPerkLevel_FieldMedic;
		var config string MinPerkLevel_Demolitionist;
		var config string MinPerkLevel_Firebug;
		var config string MinPerkLevel_Gunslinger;
		var config string MinPerkLevel_Sharpshooter;
		var config string MinPerkLevel_Survivalist;
		var config string MinPerkLevel_Swat;
		var config string DisablePerkSkills;
	/* Weapon Settings */
		var config string StartingWeapons_Berserker;
		var config string StartingWeapons_Commando;
		var config string StartingWeapons_Support;
		var config string StartingWeapons_FieldMedic;
		var config string StartingWeapons_Demolitionist;
		var config string StartingWeapons_Firebug;
		var config string StartingWeapons_Gunslinger;
		var config string StartingWeapons_Sharpshooter;
		var config string StartingWeapons_Survivalist;
		var config string StartingWeapons_Swat;
		var config string DisableWeapons;
		var config string DisableWeapons_Boss;
		var config bool bAutoAmmoBuying;
	/* Player Settings */
		var config bool bStartingWeapon_AmmoFull;
		var config bool bPlayer_SpawnWithFullArmor;
		var config bool bPlayer_SpawnWithFullGrenade;
		var config bool bEnableTraderDash;
		var config bool bDisableTeamCollisionWithTraderDash;
		var config int StartingDosh;
	/* Wave Settings */
		var config byte MaxPlayer_TotalZedsCount;
		var config byte MaxPlayer_ZedHealth;
		var config string MaxMonsters;
		var config string WaveSizeFakes;
		var config string SpawnTwoBossesName;
		var config bool bFixZedHealth_6P;
	/* ChatCommand Settings */
		var config bool bDisableChatCommand_OpenTrader;
		var config bool bDontShowOpentraderCommandInChat;
	/* */

//<<---global�ϐ��̒�`--->>//
	//WeaponConfig�N���X
		var WeaponConfig WeapCfg;
	//�E�F�[�u�^�C�v���ʂ�enum�^�̒�`
		enum eWaveType{
			WaveType_Normal,
			WaveType_Boss
		};
	//string -> byte �ϊ��p
		var array<byte> _MinPerkLevel_Berserker;
		var array<byte> _MinPerkLevel_Commando;
		var array<byte> _MinPerkLevel_Support;
		var array<byte> _MinPerkLevel_FieldMedic;
		var array<byte> _MinPerkLevel_Demolitionist;
		var array<byte> _MinPerkLevel_Firebug;
		var array<byte> _MinPerkLevel_Gunslinger;
		var array<byte> _MinPerkLevel_Sharpshooter;
		var array<byte> _MinPerkLevel_Survivalist;
		var array<byte> _MinPerkLevel_Swat;
		var array<int> _MaxMonsters;
		var array<int> _WaveSizeFakes;
	//DisablePerkSkills�p
		var array<int> aDisablePerkSkills;
		var bool bUseDisablePerkSkills;
	//SendRestrictMessageString�p
		var KFPlayerController RMPC;
		var string RMStr;
	//IsWeaponRestricted�p
		var array<string> aDisableWeapons,aDisableWeapons_Boss;
	//CheckTraderState�p
		var bool bOpened;
	//MaxPlayer_TotalZedsCount�p
		var bool bUseMaxPlayer_TotalZedsCount;
	//MaxMonsters�p
		var bool bUseMaxMonsters;
	//WaveSizeFakes�p
		var bool bUseWaveSizeFakes;
	//FillArmorOrGrenades �A�[�}�[�E�O���l�[�h�̗\���[�p
		var array<KFPawn_Human> PlayerToFillArmGre;
	//StartingDosh�\���[�p
		var array<KFPlayerReplicationInfo> PlayerToChangeStartingDosh;
	//RestrictMessage�p
		struct RestrictMessageInfo {
			var KFPlayerController KFPC;
			var string Msg;
		};
		var array<RestrictMessageInfo> RMI;
	//CheckSpawnTwoBossSquad�p
		var bool bSpawnTwoBossSquad;
	//
	
//<<---�萔�̒�`--->>//

	//�����p�[�N�g�p���̃��C�t�����l�i�g���[�_�[�^�C���j
		const VALUEFORDEAD = 10;
	//ModifyTraderTimePlayerState�p
		const TraderGroundSpeed = 364364.0f;
	//GetPerkClassFromPerkCode�p
		const PerkCode_Berserker = 0;
		const PerkCode_Commando = 1;
		const PerkCode_Support = 2;
		const PerkCode_FieldMedic = 3;
		const PerkCode_Demolitionist = 4;
		const PerkCode_Firebug = 5;
		const PerkCode_Gunslinger = 6;
		const PerkCode_Sharpshooter = 7;
		const PerkCode_Survivalist = 8;
		const PerkCode_Swat = 9;
	//
	
//<<---���b�Z�[�W�֐�--->>//

	function SetRestrictMessagePC(KFPlayerController KFPC) {
		RMPC = KFPC;
	}
	
	function SetRestrictMessageString(string s) {
		RMStr = s;
	}
	
	//�Z�b�g���ꂽ���b�Z�[�W����l�ɑ���
	function SendRestrictMessageString() {
		local string PlayerName;
		PlayerName = RMPC.PlayerReplicationInfo.PlayerName;
		ReserveRestrictMessage(RMPC,PlayerName$RMStr);
	}
	
	//���b�Z�[�W����l�ɑ���
	function SendRestrictMessageStringPC(KFPlayerController KFPC,string s) {
		SetRestrictMessagePC(KFPC);
		SetRestrictMessageString(s);
		SendRestrictMessageString();
	}
	
	//���b�Z�[�W��S���ɑ���
	function SendRestrictMessageStringAll(string s) {
		local KFPlayerController KFPC;
		foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
			ReserveRestrictMessage(KFPC,"RPWmod"$s);
		}
	}
	
	//���b�Z�[�W�̗\��
	function ReserveRestrictMessage(KFPlayerController KFPC,string s) {
		local RestrictMessageInfo addbuf;
		addbuf.KFPC = KFPC;
		addbuf.Msg = s;
		RMI.AddItem(addbuf);
		SetTimer(0.5, false, nameof(SendReserveRestrictMessageTimer));
	}
	
	//�^�C�}�[�ɂ�郁�b�Z�[�W�̔��s
	function SendReserveRestrictMessageTimer() {
		local int i;
		//���X�g���̃��b�Z�[�W���������M
			for (i=0;i<RMI.length;++i) {
				RMI[i].KFPC.TeamMessage(RMI[i].KFPC.PlayerReplicationInfo,RMI[i].Msg,'Event');
			}
		//�������I��������b�Z�[�W���폜
			RMI.Remove(0,RMI.length);
		//
	}
	
	//�R���\�[���փ��b�Z�[�W�𑗐M�v��
	function SendRestrictMessageStringConsoleAll(string s) {
//		SendConsoleMessage("RPWmod"$s);
		SendConsoleMessage(s);
	}
	
	//�R���\�[���֋�̃��b�Z�[�W�𑗐M�v��
	function SendEmptyMessageStringConsoleAll() {
		SendConsoleMessage("");
	}
	
	//�R���\�[���փ��b�Z�[�W�𑗐M�E�E�E�������������܂����x
	reliable client function SendConsoleMessage(string s) {
		SendRestrictMessageStringAll(s);
/*
		local KFGameViewportClient GVC;
		local KFPlayerController KFPC;
		local RPWPlayerController RPWPC;
		foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
			GVC = class'GameEngine'.static.GetEngine().GameViewport;
			LocalPlayer(KFPC.Player).ViewportClient.ViewportConsole.OutputText(s);
		}
*/
	}


//<<---�������֐�--->>//

	function InitConfigVar() {
		/* Init Config */
			bIWontInitThisConfig = true;
		/* Perk Settings */
			MinPerkLevel_Berserker = "0,0";
			MinPerkLevel_Commando = "0,0";
			MinPerkLevel_Support = "0,0";
			MinPerkLevel_FieldMedic = "0,0";
			MinPerkLevel_Demolitionist = "0,0";
			MinPerkLevel_Firebug = "0,0";
			MinPerkLevel_Gunslinger = "0,0";
			MinPerkLevel_Sharpshooter = "0,0";
			MinPerkLevel_Survivalist = "0,0";
			MinPerkLevel_Swat = "0,0";
			DisablePerkSkills = "";
		/* Weapon Settings */
			StartingWeapons_Berserker = "";
			StartingWeapons_Commando = "";
			StartingWeapons_Support = "";
			StartingWeapons_FieldMedic = "";
			StartingWeapons_Demolitionist = "";
			StartingWeapons_Firebug = "";
			StartingWeapons_Gunslinger = "";
			StartingWeapons_Sharpshooter = "";
			StartingWeapons_Survivalist = "";
			StartingWeapons_Swat = "";
			DisableWeapons = "";
			DisableWeapons_Boss = "";
			bAutoAmmoBuying = false;
		/* Player Settings */
			bStartingWeapon_AmmoFull = false;
			bPlayer_SpawnWithFullArmor = false;
			bPlayer_SpawnWithFullGrenade = false;
			bEnableTraderDash = false;
			bDisableTeamCollisionWithTraderDash = false;
			StartingDosh = 0;
		/* Wave Settings */
			MaxPlayer_TotalZedsCount = 0;
			MaxPlayer_ZedHealth = 0;
			MaxMonsters = "";
			WaveSizeFakes = "";
			SpawnTwoBossesName = "";
			bFixZedHealth_6P = false;
		/* ChatCommand Settings */
			bDisableChatCommand_OpenTrader = false;
			bDontShowOpentraderCommandInChat = false;
		/* */
	}
	
	//config�ϐ���������ŕێ����Ă����ϐ��ւ̕ϊ�
	function InitVarFromConfigVar() {
		SetArrayMPL(_MinPerkLevel_Berserker		,MinPerkLevel_Berserker		);
		SetArrayMPL(_MinPerkLevel_Commando		,MinPerkLevel_Commando		);
		SetArrayMPL(_MinPerkLevel_Support		,MinPerkLevel_Support		);
		SetArrayMPL(_MinPerkLevel_FieldMedic	,MinPerkLevel_FieldMedic	);
		SetArrayMPL(_MinPerkLevel_Demolitionist	,MinPerkLevel_Demolitionist	);
		SetArrayMPL(_MinPerkLevel_Firebug		,MinPerkLevel_Firebug		);
		SetArrayMPL(_MinPerkLevel_Gunslinger	,MinPerkLevel_Gunslinger	);
		SetArrayMPL(_MinPerkLevel_Sharpshooter	,MinPerkLevel_Sharpshooter	);
		SetArrayMPL(_MinPerkLevel_Survivalist	,MinPerkLevel_Survivalist)	;
		SetArrayMPL(_MinPerkLevel_Swat			,MinPerkLevel_Swat			);
		InitDisablePerkSkills(DisablePerkSkills,aDisablePerkSkills);
		InitDisableWeaponClass(DisableWeapons,aDisableWeapons);
		InitDisableWeaponClass(DisableWeapons_Boss,aDisableWeapons_Boss);
		SetArrayMM(_MaxMonsters,MaxMonsters);
		SetArrayMM(_WaveSizeFakes,WaveSizeFakes);
	}
	
	//InitVarFromConfigVar�̃T�u�֐�0
	function InitDisablePerkSkills(string DPS,out array<int> aDPS) {
		local string buf;
		local array<String> splitbuf;
		if ( DPS == "" ) return;
		bUseDisablePerkSkills = true;
		ParseStringIntoArray(DPS,splitbuf,",",true);
		foreach splitbuf(buf) {
			aDPS.AddItem(int(buf));
		}
	}
	
	//InitVarFromConfigVar�̃T�u�֐�1
	function SetArrayMPL(out array<byte> _MPL,String MPL) {
		local array<String> splitbuf;
		ParseStringIntoArray(MPL,splitbuf,",",true);
		_MPL[WaveType_Normal] = byte(splitbuf[0]);
		_MPL[WaveType_Boss] = byte(splitbuf[1]);
	};
	
	//InitVarFromConfigVar�̃T�u�֐�2
	function SetArrayMM(out array<int> _MM,String MM) {
		local string buf;
		local array<String> splitbuf;
		ParseStringIntoArray(MM,splitbuf,",",true);
		foreach splitbuf(buf) {
			_MM.AddItem(int(buf));
		}
	}
	
	//IsWeaponRestricted�p�̏������֐�
	function InitDisableWeaponClass(string StrDW,out array<string> aStrDW) {
		local string buf;
		local array<String> splitbuf;
		local class<Weapon> cWbuf;
		ParseStringIntoArray(StrDW,splitbuf,",",true);
		foreach splitbuf(buf) {
			cWbuf = GetWeapClassFromString("KFGameContent.KFWeap_" $buf);
			if (cWbuf!=None) aStrDW.AddItem(cWbuf.default.ItemName);
		}
	}

//<<---�^�C�}�[�֐�--->>//

	
	event Tick( float DeltaTime ) {
		super.Tick(DeltaTime);
		if (WeapCfg!=None) {
			if (WeapCfg.bUseWeaponConfig) WeapCfg.Tick();
		}
	}

//<<---�R�[���o�b�N�֐�(PostBeginPlay)--->>//

	//�Q�[���J�n���Ɉ�x�����Ă΂����ۂ��`
	function PostBeginPlay() {
		Super.PostBeginPlay();
		//WeaponConfig
//			WeapCfg = Spawn(class'WeaponConfig');
			WeapCfg = New(Self) class'WeaponConfig';
			WeapCfg.PostBeginPlay();
//			SetTimer(5.0f,true,nameof(test));
		//.ini�̏����ݒ�
			if (!bIWontInitThisConfig) InitConfigVar();
			SaveConfig();
			InitVarFromConfigVar();
		//CheckTraderState�p
			bOpened = true;
		//MaxPlayer_TotalZedsCount�p
			bUseMaxPlayer_TotalZedsCount = (MaxPlayer_TotalZedsCount>0);
		//MaxMonsters�p
			bUseMaxMonsters = (MaxMonsters!="");
		//WaveSizeFakes�p
			bUseWaveSizeFakes = (WaveSizeFakes!="");
		//1.0�b���Ƃɓ���̊֐����Ă� bool�l�͌J��Ԃ��ĂԂ��ǂ���
			SetTimer(1.0, true, nameof(JudgePlayers));
			SetTimer(0.25, true, nameof(CheckTraderState));
			SetTimer(1.0, true, nameof(CheckSpawnTwoBossSquad));
			SetTimer(1.0, true, nameof(HackBroadcastHandler));
			if (bAutoAmmoBuying) SetTimer(1.0, true, nameof(CheckAutoAmmoBuying));
		//
	}
	
//<<---�R�[���o�b�N�֐�(ModifyAIEnemy)--->>//
	
	/**
	
	//Controller.uc�� var Pawn Enemy
	function ModifyAIEnemy( AIController AI, Pawn Enemy ) {
		//�X�[�p�[�̏���
			super.ModifyAIEnemy(AI,Enemy);
		//
		local bool bDisableSwitchEnemysTarget;
		bDisableSwitchEnemysTarget = false;
		bDisableSwitchEnemysTarget = true;
		
		if (bDisableSwitchEnemysTarget) {
			super.ModifyAIEnemy( AI, (AI.Enemy==None) ? Enemy : AI.Enemy );
		}
	}
	
	**/
	
//<<---�R�[���o�b�N�֐�(ModifyAI)--->>//
	
	function ModifyAI(Pawn AIPawn) {
		//�X�[�p�[�̏���
			super.ModifyAI( AIPawn );
		//�ꉞ�����X�^�[������
			if (KFPawn_Monster(AIPawn)==None) return;
		//HP��ύX
			if (MaxPlayer_ZedHealth>0) {
				SetMonsterDefaultsMut(KFPawn_Monster(AIPawn),
					min( MyKFGI.GetLivingPlayerCount(), MaxPlayer_ZedHealth ));
			}
		//�Œ�HP�̕����D��x������
			if (bFixZedHealth_6P) {
				//�{�X�Ɋւ��Ă͖���
				if ( (KFPawn_ZedHans(AIPawn)!=None) || (KFPawn_ZedPatriarch(AIPawn)!=None) ) {
					return;
				}
				SetMonsterDefaultsMut(KFPawn_Monster(AIPawn),6);
			}
		//
	}
	
	//In KFGameInfo.uc function SetMonsterDefaults
	function SetMonsterDefaultsMut(KFPawn_Monster P,byte LivingPlayer) {
		local float HealthMod,HeadHealthMod;
		local int LivingPlayerCount;
		//���傫��
			LivingPlayerCount = LivingPlayer;
			HealthMod = 1.0;
			HeadHealthMod = 1.0;
		// Scale health and damage by game conductor values for versus zeds
			if( P.bVersusZed ) {
				MyKFGI.DifficultyInfo.GetVersusHealthModifier(P, LivingPlayerCount, HealthMod, HeadHealthMod);
				HealthMod *= MyKFGI.GameConductor.CurrentVersusZedHealthMod;
				HeadHealthMod *= MyKFGI.GameConductor.CurrentVersusZedHealthMod;
			}else{
				MyKFGI.DifficultyInfo.GetAIHealthModifier(P, MyKFGI.GameDifficulty, LivingPlayerCount, HealthMod, HeadHealthMod);
			}
		// Scale health by difficulty
			P.Health = P.default.Health * HealthMod;
			if( P.default.HealthMax == 0 ){
			   	P.HealthMax = P.default.Health * HealthMod;
			}else{
			   	P.HealthMax = P.default.HealthMax * HealthMod;
			}
			P.ApplySpecialZoneHealthMod(HeadHealthMod);
			P.GameResistancePct = MyKFGI.DifficultyInfo.GetDamageResistanceModifier(LivingPlayerCount);
		//
	}


//<<---�R�[���o�b�N�֐�(ModifyPlayer)--->>//

	//�v���C���[���������ꂽ�Ƃ��Ɉ�x�����Ă΂��
	function ModifyPlayer(Pawn Other) {
		local KFPawn Player;
		local KFPlayerController KFPC;
		local KFPlayerReplicationInfo KFPRI;
		local class<KFPerk> cKFP;
		local class<Weapon> cRetW;
		local Inventory Inv;
		local bool bFound;
		//�X�[�p�[�̏���
			super.ModifyPlayer(Other);
		//���
			Player = KFPawn_Human(Other);
			KFPC = KFPlayerController(Player.Controller);
			cKFP = KFPC.GetPerk().GetPerkClass();
			KFPRI = KFPlayerReplicationInfo(Player.PlayerReplicationInfo);
		//SpawnHumanPawn�Ɋւ��Ă͖���
			if (IsBotPlayer(KFPC)) return;
		//���������̕ύX��̏����擾 �ύX�����ꍇ�̂ݏ���
			cRetW = GetStartingWeapClassFromPerk(cKFP);
			if (cRetW!=None) {
				//�v���C���[���珉���������͂��D �T�o�C�o���X�g���l�����đS�p�[�N����Ŕ���
					bFound = False;
					for(Inv=Player.InvManager.InventoryChain;Inv!=None;Inv=Inv.Inventory) {
						switch(Inv.ItemName) {
							case class'KFGameContent.KFWeap_Blunt_Crovel'.default.ItemName:
							case class'KFGameContent.KFWeap_AssaultRifle_AR15'.default.ItemName:
							case class'KFGameContent.KFWeap_Shotgun_MB500'.default.ItemName:
							case class'KFGameContent.KFWeap_Pistol_Medic'.default.ItemName:
							case class'KFGameContent.KFWeap_GrenadeLauncher_HX25'.default.ItemName:
							case class'KFGameContent.KFWeap_Flame_CaulkBurn'.default.ItemName:
							case class'KFGameContent.KFWeap_Revolver_DualRem1858'.default.ItemName:
							case class'KFGameContent.KFWeap_Rifle_Winchester1894'.default.ItemName:
							case class'KFGameContent.KFWeap_SMG_MP7'.default.ItemName:
								Player.InvManager.RemoveFromInventory(Inv);
								bFound = True;
								break;
						}
						if (bFound) break;
					}
				//����̂Ɓ[�ɂ�[
					Player.Weapon = Weapon(Player.CreateInventory(cRetW,Player.Weapon!=None));
				//����̑���
					Player.InvManager.ServerSetCurrentWeapon(Player.Weapon);
				//�e��̕�[ 
					if (bStartingWeapon_AmmoFull) FillWeaponAmmo(KFWeapon(Player.Weapon));
				//�A�[�}�[�E�O���l�[�h�̕�[�\�� �Ώ��Ö@�I�Ȃ̂Ńo�O��\�������邪�c �����Ȃ����ꍇ��KFPerk����Init��j�~���邵���Ȃ��Ȃ�
					PlayerToFillArmGre.AddItem(KFPawn_Human(Other));
					SetTimer(0.25, false, nameof(FillArmorOrGrenades)); //�Ăяo����1��Ȃ̂�false
				//
			}
		//�J�n����Dosh�̐ݒ�
			if (StartingDosh>0) {
				PlayerToChangeStartingDosh.AddItem(KFPRI);
				SetTimer(0.25, false, nameof(SetStartingDosh)); //�Ăяo����1��Ȃ̂�false
			}
		//�����I��
	}
	
	//StartingDosh�̗\��ύX
	function SetStartingDosh() {
		local int i;
		//�ύX�Ώێ҂̃��X�g������
			for (i=0;i<PlayerToChangeStartingDosh.length;++i) {
				if (PlayerToChangeStartingDosh[i]!=None) {
					PlayerToChangeStartingDosh[i].Score = StartingDosh;
				}
			}
		//�\�񃊃X�g�̏�����
			PlayerToChangeStartingDosh.Remove(0,PlayerToChangeStartingDosh.length);
		//
	}
	
	//����₭�ق���[
	function FillWeaponAmmo(KFWeapon W) {
		if ( W.AddAmmo(114514) != 0 ) FillWeaponAmmo(W);
		if ( W.AddSecondaryAmmo(114514) != 0 ) FillWeaponAmmo(W);
	}
	
	//�O���ƃA�[�}�[�́i�\��j�ق���[
	function FillArmorOrGrenades() {
		local KFPawn_Human Player;
		local int i;
		//��[�Ώێ҂̃��X�g������
			for (i=0;i<PlayerToFillArmGre.length;++i) {
				Player = PlayerToFillArmGre[i];
				if (Player==None) continue;
				//�A�[�}�[�̕�[
					if (bPlayer_SpawnWithFullArmor) Player.GiveMaxArmor();
				//�O���l�[�h�̕�[
					if (bPlayer_SpawnWithFullGrenade) FillGrenades(KFInventoryManager(Player.InvManager));
				//
			}
		//�������I������v���C���[���폜
			PlayerToFillArmGre.Remove(0,PlayerToFillArmGre.length);
		//
	}
	
	//�O����[�p
	function FillGrenades(KFInventoryManager KFIM) {
		if (KFIM.AddGrenades(1)) FillGrenades(KFIM);
	}
	
	//Are you SpawnHumanPawn?
	function bool IsBotPlayer(KFPlayerController KFPC){
		return (KFPC.PlayerReplicationInfo.PlayerName=="Braindead Human");
	}

//<<---�J�n����̏�����--->>//
	
	//���݂̃p�[�N����N���X�̎擾�B
	function class<Weapon> GetStartingWeapClassFromPerk(class<KFPerk> Perk) {
		local string SendStr;
		local array<String> SplitBuf;
		SendStr = "";
		switch(Perk) {
			case class'KFPerk_Berserker':
				SendStr = StartingWeapons_Berserker;
				break;
			case class'KFPerk_Commando':
				SendStr = StartingWeapons_Commando;
				break;
			case class'KFPerk_Support':
				SendStr = StartingWeapons_Support;
				break;
			case class'KFPerk_FieldMedic':
				SendStr = StartingWeapons_FieldMedic;
				break;
			case class'KFPerk_Demolitionist':
				SendStr = StartingWeapons_Demolitionist;
				break;
			case class'KFPerk_Firebug':
				SendStr = StartingWeapons_Firebug;
				break;
			case class'KFPerk_Gunslinger':
				SendStr = StartingWeapons_Gunslinger;
				break;
			case class'KFPerk_Sharpshooter':
				SendStr = StartingWeapons_Sharpshooter;
				break;
			case class'KFPerk_Survivalist':
				SendStr = StartingWeapons_Survivalist;
				break;
			case class'KFPerk_Swat':
				SendStr = StartingWeapons_Swat;
				break;
		}
		if (SendStr=="") return None;
		ParseStringIntoArray(SendStr,SplitBuf,",",true);
		SendStr = "KFGameContent.KFWeap_" $ SplitBuf[Rand(SplitBuf.length)];
		return GetWeapClassFromString(SendStr);
	}
	
	//�����񂩂�N���X�̎擾�B
	function class<Weapon> GetWeapClassFromString(string str) {
		return class<Weapon>(DynamicLoadObject(str, class'Class'));
	}
	
//<<---���C���֐�(CheckSpawnTwoBossSquad)--->>//
	
	//2�̖ڂ̃{�X������ 1.0�b���ƂɌĂяo��
	function CheckSpawnTwoBossSquad() {
		local byte Curwave;
		//������
			Curwave = MyKFGI.MyKFGRI.WaveNum;
		//�E�F�[�u���J�n����Ă��Ȃ��ꍇ�͂ǂ��ł�����
			if (!(Curwave>=1)) return;
//�����I�Ƀ{�X�E�F�[�u�ɂ���e�X�g�R�[�h
//		if (Curwave<MyKFGI.MyKFGRI.WaveMax) KFGameInfo_Survival(MyKFGI).WaveEnded(WEC_WaveWon);
		//�{�X2�̂̏�������
			if (Curwave==MyKFGI.MyKFGRI.WaveMax) {
				if (bSpawnTwoBossSquad) {
					MyKFGI.SpawnManager.TimeUntilNextSpawn = 10;
					SetTimer(5.0, false, nameof(SpawnTwoBosses));
					SetTimer(15.0, false, nameof(SpawnTwoBosses));
					bSpawnTwoBossSquad = false;
				}
			}else{
				bSpawnTwoBossSquad = true;
			}
		//
	}
	
	function SpawnTwoBosses() {
		local array<class<KFPawn_Monster> > SpawnList;
		local array<String> SplitBuf;
		local string Buf;
		//Add 2018.02.11
			local array<class<KFPawn_Monster> > Bosses_Class;
			local array<String> Bosses_Name;
			local byte i,Bosses_Len;
			Bosses_Name.AddItem("Hans");	Bosses_Class.AddItem(class'KFPawn_ZedHans');
			Bosses_Name.AddItem("Pat");		Bosses_Class.AddItem(class'KFPawn_ZedPatriarch');
			Bosses_Name.AddItem("KFP");		Bosses_Class.AddItem(class'KFPawn_ZedFleshpoundKing');
			Bosses_Name.AddItem("KBlt");	Bosses_Class.AddItem(class'KFPawn_ZedBloatKing');
			Bosses_Len = Bosses_Name.Length;
		//
		if (SpawnTwoBossesName=="") {
			MyKFGI.SpawnManager.TimeUntilNextSpawn = 0;
			return;
		}
		ParseStringIntoArray(SpawnTwoBossesName,SplitBuf,",",true);
		foreach SplitBuf(Buf) {
			//Change 2018.02.11
				if (Buf=="Rand") {
					SpawnList.AddItem(Bosses_Class[Rand(Bosses_Len)]);
				}else{
					for(i=0;i<Bosses_Len-1;i++) {
						if (Buf==Bosses_Name[i]) SpawnList.AddItem(Bosses_Class[i]);
					}
				}
//SendRestrictMessageStringAll(Buf);	//test 
		}
		MyKFGI.NumAISpawnsQueued += MyKFGI.SpawnManager.SpawnSquad( SpawnList );
		MyKFGI.SpawnManager.TimeUntilNextSpawn = MyKFGI.SpawnManager.CalcNextGroupSpawnTime();
//	SendRestrictMessageStringAll("�L���[�̐��F" $ MyKFGI.NumAISpawnsQueued $ "  ___  " $ MyKFGI.SpawnManager.TimeUntilNextSpawn	);
	}

//<<---���C���֐�(CheckAutoAmmoBuying)--->>//

	//�S�v���C���[�ɂ��Ēe��̕�[���m�F����
	function CheckAutoAmmoBuying() {
		local KFPlayerController KFPC;
		//�E�F�[�u���J�n����Ă��Ȃ��ꍇ�͂ǂ��ł�����
			if (!(MyKFGI.MyKFGRI.WaveNum>=1)) return;
		//�SPC�ɑ΂��Ĕ��� SpawnHumanPawn�Ɋւ��Ă͖���
			foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
				if ( (KFPC!=None) && (!IsBotPlayer(KFPC)) ) {
					AutoAmmoBuy(KFPC);
				}
			}
		//
	}
	
	//�e�򎩓��`���[�W 2018.02.10
	function AutoAmmoBuy(KFPlayerController KFPC) {
		local KFPlayerReplicationInfo KFPRI;
		local KFWeapon KFWeap;
		local int BASH_FIREMODE,i,price,WeapMC;
		local class<KFWeaponDefinition> WDClass;
		local bool bEvisAlt;
		BASH_FIREMODE = 3;
		KFPRI = KFPlayerReplicationInfo(KFPC.PlayerReplicationInfo);
		KFWeap = KFWeapon(KFPC.Pawn.Weapon);
		bEvisAlt = false;
		if (KFWeap==None) return;
		//���C���E�T�u�}�K�W�����ꂼ��ɂ��Č���
		for (i=0;i<2;i++) {
			//�\���e�q��2�}�K�W����؂�����w��
			WeapMC = KFWeap.MagazineCapacity[i];
			if (KFWeap.SpareAmmoCount[i] < 2 * WeapMC) {
				//�e��̉��i�𒲂ׂ������AKFWeap����KFWeapDef�𒼐ڎ��ɍs���Ȃ��̂ŁAKFDT���o�R
					WDClass = class<KFDamageType>(KFWeap.class.default.InstantHitDamageTypes[BASH_FIREMODE]).default.WeaponDef;
					price = (i==0 ? WDClass.default.AmmoPricePerMag : WDClass.default.SecondaryAmmoMagPrice);
					//�}�K�W�����ʂɑΉ� �������Ă��ꍇ���̕������Ƃ낤��
						if (KFWeap.class.default.MagazineCapacity[i] != 0 ) {
							price = (price * KFWeap.MagazineCapacity[i]) / KFWeap.class.default.MagazineCapacity[i];
						}
					//
					if (price<=0) continue; //�����Ȃ����̂�����
					if (KFPRI.Score<price) continue; //����������Ȃ���I
				//�����Ɨ\���e�q�n�~��ꍇ�͋���
					if (KFWeap.SpareAmmoCapacity[i] < KFWeap.SpareAmmoCount[i] + WeapMC) {
						//�G���B�X����̃I���g�t�@�C�A�͕ʂȂ񂾁c�c
						if ( (KFWeap_Eviscerator(KFWeap)!=None) && (i==1) ) {
							WeapMC = WDClass.default.SecondaryAmmoMagSize; //10��������
							bEvisAlt = ( KFWeap.AmmoCount[1] < 2* WeapMC ); //�I���g�̒e�q��20��؂��Ă�Ȃ�w��
						}
						if (!bEvisAlt) continue;
					}
				//��[�ƍw������
//SendRestrictMessageStringAll("less::" $ (i==0?"Main":"Sub"));
					if (bEvisAlt) {
						KFWeap.AmmoCount[1] += WeapMC;
					}else{
						KFWeap.SpareAmmoCount[i] += WeapMC;
					}
					KFPRI.Score -= price;
				//
			}
		}
//		KFWeap.SpareAmmoCount[0] = KFWeap.SpareAmmoCapacity[0];
//		KFWeap.SpareAmmoCapacity[0] = 99999;
//		KFWeap.SpareAmmoCount[1] = KFWeap.SpareAmmoCapacity[1];
	}

//<<---���C���֐�(CheckTraderState)--->>//
	
	function CheckTraderState() {
		local byte PlayerCount;
		//�E�F�[�u���J�n����Ă��Ȃ��ꍇ�͂ǂ��ł�����
			if (!(MyKFGI.MyKFGRI.WaveNum>=1)) return;
		//���X�����܂�܂����`��
			if ( (MyKFGI.MyKFGRI.bTraderIsOpen==false) && (bOpened==true) ) {
				//TotalAICount,Maxmonsters ���Judge�֐������Ŏ��l���̕��ׂ��p�[�e�B�ɂ����Ȃ��悤�ɂ���
					JudgePlayers();
					PlayerCount = MyKFGI.GetLivingPlayerCount();
					if (PlayerCount>0) {
						//bosswave�ȊO�ł̂ݎ��s
							if (MyKFGI.MyKFGRI.WaveNum<MyKFGI.MyKFGRI.WaveMax) {
								//TotalZed�����炷
									if (bUseMaxPlayer_TotalZedsCount) SetMaxPlayer_TotalZedsCount(PlayerCount);
								//WaveSizeFakes
									if (bUseWaveSizeFakes) SetWaveSizeFakes(PlayerCount);
								//
							}
						//������������ύX
							if (bUseMaxMonsters) SetCustomMaxMonsters(PlayerCount);
						//
					}
				//���鑬�x�����ɖ߂�
					ModifyTraderTimePlayerState(false);
				//
			}
		//���X�������Ă܂��`��
			if ( MyKFGI.MyKFGRI.bTraderIsOpen==true ) {
				ModifyTraderTimePlayerState(true);
			}
		//��Ԃ̕ۑ�
			bOpened = MyKFGI.MyKFGRI.bTraderIsOpen;
		//
	}
	
	//WaveSizeFakes�̐ݒ�
	function SetWaveSizeFakes(byte PlayerCount) {
		local byte WSF;
//		WSF = _WaveSizeFakes[min( PlayerCount-1+30, _WaveSizeFakes.length-1 )];
//		SetCustomTotalAICount(PlayerCount+WSF,true);
		WSF = _WaveSizeFakes[min( PlayerCount-1, _WaveSizeFakes.length-1 )];
		if (WSF>0) {
			SetCustomTotalAICount(PlayerCount+WSF,false);
			SendRestrictMessageStringAll("::SetWaveSizeFakes "$WSF);
		}
	}
	
	//MaxPlayer_TotalZedsCount�̐ݒ� �l���ߑ��̏ꍇ���炷
	function SetMaxPlayer_TotalZedsCount(byte PlayerCount) {
		if (PlayerCount>MaxPlayer_TotalZedsCount) {
			SetCustomTotalAICount(MaxPlayer_TotalZedsCount,true);
		}
	}

	//�E�F�[�u�ŕ���MOB���̒��� �Q�l�� 'KFAISpawnManager.uc' func: SetupNextWave
	function SetCustomTotalAICount(byte PlayerCount,bool bOutPutLog) {
		local int OldAIcount,AIcount;
		OldAIcount = MyKFGI.SpawnManager.WaveTotalAI;
		AIcount = 	MyKFGI.SpawnManager.WaveSettings.Waves[ MyKFGI.MyKFGRI.WaveNum-1 ].MaxAI *
					MyKFGI.DifficultyInfo.GetPlayerNumMaxAIModifier( PlayerCount ) *
					MyKFGI.DifficultyInfo.GetDifficultyMaxAIModifier();
		MyKFGI.SpawnManager.WaveTotalAI = AIcount;
		MyKFGI.MyKFGRI.AIRemaining = AIcount;
		MyKFGI.MyKFGRI.WaveTotalAICount = AIcount;
		if (bOutPutLog) SendRestrictMessageStringAll("::SetTotalZedsCount "$OldAIcount$"->"$AIcount);
	}
	
	//�����������̒���
	function SetCustomMaxMonsters(byte PlayerCount) {
		local int MaxZeds;
		MaxZeds = min ( 255, _MaxMonsters[min( PlayerCount-1, _MaxMonsters.length-1 )] );
//this is old ver
//		MyKFGI.SpawnManager.MaxMonstersSolo[int(MyKFGI.GameDifficulty)] = byte(MaxZeds);
//		MyKFGI.SpawnManager.MaxMonsters = byte(MaxZeds);
//for v1056
		SetMaxMonstersV1056(byte(MaxZeds));
//
		SendRestrictMessageStringAll("::SetMaxMonsters "$byte(MaxZeds));
	}
	
	//MM�̐ݒ� - KFmutator_MaxplayersV2���R�s�y v1056����Փx�E�l������MM���ł����悤��
	function SetMaxMonstersV1056(byte mm_v1056) {
		local int i,j;
		for (i = 0; i < MyKFGI.SpawnManager.PerDifficultyMaxMonsters.length; i++) {
			for (j = 0; j < MyKFGI.SpawnManager.PerDifficultyMaxMonsters[i].MaxMonsters.length ; j++) {
				MyKFGI.SpawnManager.PerDifficultyMaxMonsters[i].MaxMonsters[j] = mm_v1056;
			}
		}
	}
	
	//�g���[�_�[�J�X�y�ѕX���̃v���C���[�̏�Ԃ̕ύX
	function ModifyTraderTimePlayerState(bool bOpenTrader) {
		local KFPlayerController KFPC;
		local KFPawn Player;
/*		//�g���[�_�[���̃p�[�N�ύX�Ɋւ���ݒ�
			if (bAllowChangingPerkAnytimeInTraderTime && bOpenTrader ) {
					foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
						if (KFPC!=None) KFPC.SetHaveUpdatePerk(false);
//KFPC.TeamMessage(KFPC.PlayerReplicationInfo,"No,match::?"$KFGameReplicationInfo(KFPC.WorldInfo.GRI).bMatchHasBegun,'Event');
					}
				}
			}*/
//
		//�g���[�_�[�_�b�V���Ɋւ���ݒ�
			if (bEnableTraderDash) {
				foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
					Player = KFPawn(KFPC.Pawn);
					if (Player!=None) {
						SetCustomSpeedAndCollision(Player,bOpenTrader);
					}
				}
			}
		//
	}
	
	//�ړ����x�ƃR���W�����̕ύX
	function SetCustomSpeedAndCollision(KFPawn Player,bool bOpenTrader) {
		//�X�s�[�h�̕ύX (�i�C�t�������Ă鎞)
			if ( bOpenTrader && IsPlayerKnifeOut(Player) ) {
				Player.GroundSpeed = TraderGroundSpeed;
			}else{
				Player.UpdateGroundSpeed();
			}
		//�R���W�����̗L��
			if (bDisableTeamCollisionWithTraderDash) {
				Player.bIgnoreTeamCollision = bOpenTrader ? true : MyKFGI.bDisableTeamCollision;
			}
		//
	}

	//�i�C�t�������Ă��邩
	function bool IsPlayerKnifeOut(KFPawn Player) {
		return (KFWeap_Edged_Knife(Player.Weapon)!=None);
	}

//<<---���C���֐�(JudgePlayers)--->>//

	
	//�v���C���[�ɍق����I
	function JudgePlayers() {
		local KFPlayerController KFPC;
		local eWaveType eWT;
		//�E�F�[�u���J�n����Ă��Ȃ��ꍇ�͂ǂ��ł�����
			if (!(MyKFGI.MyKFGRI.WaveNum>=1)) return;
		//���݃E�F�[�u���ʏ킩�{�X�� WaveNum�����݂̃E�F�[�u�AWaveMax�͍ő�E�F�[�u
			eWT =  ( MyKFGI.MyKFGRI.WaveNum < MyKFGI.MyKFGRI.WaveMax - ( MyKFGI.MyKFGRI.bTraderIsOpen ? 1 : 0 ) ) ? WaveType_Normal : WaveType_Boss;
		//�SPC�ɑ΂��Ĕ��� SpawnHumanPawn�Ɋւ��Ă͖���
			foreach WorldInfo.AllControllers(class'KFPlayerController', KFPC) {
				if ( (KFPC!=None) && (!IsBotPlayer(KFPC)) ) {
					JudgeSpecificPlayer(KFPC,eWT);
				}
			}
		//
	}
	
	//JudgePlayers�̃T�u�֐�
	function JudgeSpecificPlayer(KFPlayerController KFPC,eWaveType eWT) {
		local Pawn Player;
		local Weapon CurWeapon;
		//���݂��Ȃ��v���C���[���
			if (KFPC.Pawn==None) return;
		//������
			Player = KFPC.Pawn;
			CurWeapon = Player.Weapon;
		//���b�Z�[�W�̑��M������̃v���C���[�ɐݒ�
			SetRestrictMessagePC(KFPC);
		//�֎~������g�p���Ă���ꍇ�͋����I�ɏ��ł����� Old: KFPC.ServerThrowOtherWeapon(CurWeapon);
			if (IsWeaponRestricted(CurWeapon,eWT)) {
				SendRestrictMessageString();
				CurWeapon.Destroyed();
			}
		//�p�[�N�̃��x������or�X�L������	�g���[�_�[���J���Ă���Ȃ�HP�����炷 �����łȂ���ΎE��
			if ( IsPerkLevelRestricted(KFPC.GetPerk().GetPerkClass(), KFPC.GetPerk().GetLevel(),eWT)
																|| IsUsingRestrictedPerkSkill(KFPC,eWT) ) {
				if (MyKFGI.MyKFGRI.bTraderIsOpen) {
					Player.Health = max(Player.Health-VALUEFORDEAD,1);
				}else if (Player.Health>=0) {
					SendRestrictMessageString();
					Player.FellOutOfWorld(none);
				}
			}
		//
	}

//<<---��������֐�--->>//
		
	//�p�[�N���x���������𖞂����Ă��邩�ǂ���
	function bool IsPerkLevelRestricted(class<KFPerk> Perk,byte PerkLevel,eWaveType eWT) {
		switch(Perk) {
			case class'KFPerk_Berserker':
				return IsBadPerkLevel(_MinPerkLevel_Berserker[eWT],PerkLevel,Perk);
			case class'KFPerk_Commando':
				return IsBadPerkLevel(_MinPerkLevel_Commando[eWT],PerkLevel,Perk);
			case class'KFPerk_Support':
				return IsBadPerkLevel(_MinPerkLevel_Support[eWT],PerkLevel,Perk);
			case class'KFPerk_FieldMedic':
				return IsBadPerkLevel(_MinPerkLevel_FieldMedic[eWT],PerkLevel,Perk);
			case class'KFPerk_Demolitionist':
				return IsBadPerkLevel(_MinPerkLevel_Demolitionist[eWT],PerkLevel,Perk);
			case class'KFPerk_Firebug':
				return IsBadPerkLevel(_MinPerkLevel_Firebug[eWT],PerkLevel,Perk);
			case class'KFPerk_Gunslinger':
				return IsBadPerkLevel(_MinPerkLevel_Gunslinger[eWT],PerkLevel,Perk);
			case class'KFPerk_Sharpshooter':
				return IsBadPerkLevel(_MinPerkLevel_Sharpshooter[eWT],PerkLevel,Perk);
			case class'KFPerk_Survivalist':
				return IsBadPerkLevel(_MinPerkLevel_Survivalist[eWT],PerkLevel,Perk);
			case class'KFPerk_Swat':
				return IsBadPerkLevel(_MinPerkLevel_Swat[eWT],PerkLevel,Perk);
		}
		return false;
	}
	
	//IsPerkLevelRestricted�̃T�u�֐�
	function bool IsBadPerkLevel(byte MinPerkLevel,byte PerkLevel,class<KFPerk> Perk) {
		local string perkname;
		if (!(MinPerkLevel<=PerkLevel)) {
			perkname = Perk.default.PerkName;
			if (MinPerkLevel<=25) {
				SetRestrictMessageString("::FellOutOfWorld::"$perkname$"::NeedLevel"$MinPerkLevel$"(You:"$PerkLevel$")");
			}else{
				SetRestrictMessageString("::FellOutOfWorld::"$perkname$"::RestrictedPerk");
			}
			return true;
		}else{
			return false;
		}
	}
		
//----------------testcodes----------------//

	//�g�p�֎~���킩�ǂ���
	function bool IsWeaponRestricted(Weapon Weap,eWaveType eWT) {
		local array<string> aDWName;
		local string WName,DWName;
		//�O����
			if (KFWeapon(Weap)==None) return false;
			WName = Weap.ItemName;
			if (eWT==WaveType_Normal)	aDWName = aDisableWeapons;
			if (eWT==WaveType_Boss)		aDWName = aDisableWeapons_Boss;
		//�e���했�̔���B
			foreach aDWName(DWName) {
				if (WName==DWName) {
					SetRestrictMessageString("::DestroyWeapon::"$WName$"::RestrictedWeapon");
					return true;
				}
			}
		//
		return false;
	}
	
	//�g�p�֎~�X�L�����g���Ă��邩�ǂ���
	function bool IsUsingRestrictedPerkSkill(KFPlayerController KFPC,eWaveType eWT) {
		local int val,wavecodeongame,PerkCode,SkillCode,WaveCode;
		local string skillinfo;
		if (!bUseDisablePerkSkills) return false;
		if (eWT==WaveType_Normal) wavecodeongame = 0;
		if (eWT==WaveType_Boss) wavecodeongame = 1;
		foreach aDisablePerkSkills(val) {
			PerkCode	= (val/100)	%10;
			SkillCode	= (val/10)	%10;
			WaveCode	= (val)		%10;
			//PerkCode SkillCode WaveCode
			if ( ( GetPerkClassFromPerkCode(PerkCode) == KFPC.GetPerk().GetPerkClass() ) &&
				 ( KFPC.GetPerk().PerkSkills[SkillCode].bActive ) &&
				 ( WaveCode == wavecodeongame ) ) {
				skillinfo = GetSkillInfo(KFPC.GetPerk().GetPerkClass(),SkillCode);
				SetRestrictMessageString("::FellOutOfWorld::"$skillinfo$"::RestrictedSkill");
				return true;
			}
		}
		return false;
	}
	
	//�X�L���̏��
	function string GetSkillInfo(class<KFPerk> Perk,byte SkillCode) {
		local string AdditionalInfo;
		AdditionalInfo = "(Lv"$(5*((SkillCode/2)+1));
		AdditionalInfo $= ( SkillCode%2 == 0 ) ? "L" : "R";
		AdditionalInfo $= ")";
		return ( Perk.default.PerkSkills[SkillCode].Name $ AdditionalInfo );
	}
	
	//PerkCode����Perk�N���X���擾
	function class<KFPerk> GetPerkClassFromPerkCode(byte pcode) {
		switch(pcode) {
			case PerkCode_Berserker:
				return class'KFPerk_Berserker';
			case PerkCode_Commando:
				return class'KFPerk_Commando';
			case PerkCode_Support:
				return class'KFPerk_Support';
			case PerkCode_FieldMedic:
				return class'KFPerk_FieldMedic';
			case PerkCode_Demolitionist:
				return class'KFPerk_Demolitionist';
			case PerkCode_Firebug:
				return class'KFPerk_Firebug';
			case PerkCode_Gunslinger:
				return class'KFPerk_Gunslinger';
			case PerkCode_Sharpshooter:
				return class'KFPerk_Sharpshooter';
			case PerkCode_Survivalist:
				return class'KFPerk_Survivalist';
			case PerkCode_Swat:
				return class'KFPerk_Swat';
		}
	}
	

//<<---�`���b�g�R�}���h�֘A(ChatCommands)--->>//

	//chat�̏�����
	function HackBroadcastHandler() {
		if (RPWBroadcastHandler(MyKFGI.BroadcastHandler)==None) {
			MyKFGI.BroadcastHandler = spawn(class'RPWBroadcastHandler');
			RPWBroadcastHandler(MyKFGI.BroadcastHandler).InitRPWClass(Self);
		 	ClearTimer(nameof(HackBroadcastHandler));
		}
	}
	
	//PRI����KFPC�̎擾
	function KFPlayerController GetKFPCFromPRI(PlayerReplicationInfo PRI) {
		return KFPlayerController(KFPlayerReplicationInfo(PRI).Owner);
	}
	
	//chat���e�̃t�b�N �Ԃ�l�͂��̃e�L�X�g���\���ɂ��邩�ǂ���
	function Broadcast(PlayerReplicationInfo SenderPRI,coerce string Msg) {
		local string MsgHead,MsgBody;
		local array<String> splitbuf;
		//split message:
			ParseStringIntoArray(Msg,splitbuf," ",true);
			MsgHead = splitbuf[0];
			MsgBody = splitbuf[1];
		//���b�Z�[�W���e�ŕ���
			switch(MsgHead) {
				case "!rpwdebug":
					Broadcast_Debug(GetKFPCFromPRI(SenderPRI));
					break;
				case "!OpenTrader":
				case "!OT":
					if (bDisableChatCommand_OpenTrader) break;
					if (MsgBody=="") Broadcast_OpenTrader(GetKFPCFromPRI(SenderPRI));
					break;
				case "!WaveSizeFakes":
					Broadcast_WaveSizeFakes(MsgBody);
					break;
				case "!RPWInfo":
					if (MsgBody=="") Broadcast_RPWInfo();
					break;
				case "!hawawa":
					Broadcast_Special(MsgBody);
					break;
			}
		//
	}
	
	function bool StopBroadcast(string Msg) {
		local string MsgHead,MsgBody;
		local array<String> splitbuf;
		//split message:
			ParseStringIntoArray(Msg,splitbuf," ",true);
			MsgHead = splitbuf[0];
			MsgBody = splitbuf[1];
		//"!opentrader"��"!ot"�̏���
			switch(MsgHead) {
				case "!OpenTrader":
				case "!OT":
					if (MsgBody=="") return bDontShowOpentraderCommandInChat;
					break;
			}
		//�ʏ�͂��̂܂܃e�L�X�g��\��
			return false;
		//
	}
	
	//!rpwdebug: mod�쐬�⏕
	function Broadcast_Debug(KFPlayerController KFPC) {
	}
	
	//!OpenTrader: �����J�X
	function Broadcast_OpenTrader(KFPlayerController KFPC) {
		if (MyKFGI.MyKFGRI.bTraderIsOpen) KFPC.OpenTraderMenu();
	}
	
	//!WaveSizeFakes: WSF�̋@�\�̎g�p�؂�ւ�
	function Broadcast_WaveSizeFakes(string Msg) {
		if (Msg=="true") {
			bUseWaveSizeFakes = true;
			SendRestrictMessageStringAll("::WaveSizeFakes::Enable");
		}
		if (Msg=="false") {
			bUseWaveSizeFakes = false;
			SendRestrictMessageStringAll("::WaveSizeFakes::Disable");
		}
	}
	
	//!RPWInfo: RPWmod�̏��
	function Broadcast_RPWInfo() {
		local string InfoBuf;
		//���̑��M�J�n
//			SendRestrictMessageStringAll("::RPWInfo in console.");
//			SendEmptyMessageStringConsoleAll();
			SendRestrictMessageStringConsoleAll("::RPWInfo");
		//�p�[�N���x��
			InfoBuf $= "MinPerkLevel(1w-10w,Boss)// ";
			Broadcast_RPWInfo_AddPerkInfo(InfoBuf);
			SendRestrictMessageStringConsoleAll("::"$InfoBuf);
		//�֎~�X�L��
			if (bUseDisablePerkSkills) {
				InfoBuf = "DisableSkills// ";
				Broadcast_RPWInfo_AddSkillsInfo(InfoBuf);
				SendRestrictMessageStringConsoleAll("::"$InfoBuf);
			}
		//�֎~����
			if (DisableWeapons!="") {
				InfoBuf = "DisableWeap// ";
				Broadcast_RPWInfo_AddWeapInfo(InfoBuf,aDisableWeapons);
				SendRestrictMessageStringConsoleAll("::"$InfoBuf);
			}
		//�֎~����Boss
			if (DisableWeapons_Boss!="") {
				InfoBuf = "DisableWeap(Boss)// ";
				Broadcast_RPWInfo_AddWeapInfo(InfoBuf,aDisableWeapons_Boss);
				SendRestrictMessageStringConsoleAll("::"$InfoBuf);
			}
		//�����{�X�̖��O
			if (SpawnTwoBossesName!="") {
				InfoBuf = "BossName// "$SpawnTwoBossesName;
				SendRestrictMessageStringConsoleAll("::"$InfoBuf);
			}
		//
	}
	
	//RPWInfo�Ƀp�[�N���x���̏�����������
	function Broadcast_RPWInfo_AddPerkInfo(out string InfoBuf) {
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Berserker)		$"("$MinPerkLevel_Berserker$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Commando)		$"("$MinPerkLevel_Commando$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Support)		$"("$MinPerkLevel_Support$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_FieldMedic)		$"("$MinPerkLevel_FieldMedic$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Demolitionist)	$"("$MinPerkLevel_Demolitionist$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Firebug)		$"("$MinPerkLevel_Firebug$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Gunslinger)		$"("$MinPerkLevel_Gunslinger$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Sharpshooter)	$"("$MinPerkLevel_Sharpshooter$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Survivalist)	$"("$MinPerkLevel_Survivalist$") ";
		InfoBuf $= GetPerkNameFromPerkCode(PerkCode_Swat)			$"("$MinPerkLevel_Swat$") ";
	}
	
	//PerkCode����i�Ǝ��́j�p�[�N�̖��O���擾
	function string GetPerkNameFromPerkCode(int PerkCode) {
		switch(PerkCode) {
			case PerkCode_Berserker:
				return "Zerk";
			case PerkCode_Commando:
				return "Com";
			case PerkCode_Support:
				return "Sup";
			case PerkCode_FieldMedic:
				return "Med";
			case PerkCode_Demolitionist:
				return "Demo";
			case PerkCode_Firebug:
				return "Bug";
			case PerkCode_Gunslinger:
				return "GS";
			case PerkCode_Sharpshooter:
				return "SS";
			case PerkCode_Survivalist:
				return "Suv";
			case PerkCode_Swat:
				return "Swat";
		}
		return "NullPerk?[ERROR@GetPerkNameFromPerkCode]";
	}

	//RPWInfo�ɋ֎~�X�L���̏�����������
	function Broadcast_RPWInfo_AddSkillsInfo(out string InfoBuf) {
		local bool nl; //new line
		local int val,PerkCode,SkillCode,WaveCode;
		nl = false;
		foreach aDisablePerkSkills(val) {
			//�O����
				PerkCode	= (val/100)	%10;
				SkillCode	= (val/10)	%10;
				WaveCode	= (val)		%10;
			//Info��������
				if (nl) InfoBuf $= " ";
//				InfoBuf $= "[";
				InfoBuf $= GetPerkNameFromPerkCode(PerkCode)$"[";
				InfoBuf $= GetSkillInfo(GetPerkClassFromPerkCode(PerkCode),SkillCode)$"]";
				if (wavecode==1) InfoBuf $= "(Boss)";
//				InfoBuf $= "]";
			//
			nl = true;
		}
	}
	
	//RPWInfo�ɕ���̖��O������
	function Broadcast_RPWInfo_AddWeapInfo(out string InfoBuf,array<string> aWeapName) {
		local string WName;
		local bool nl; //new line
		nl = false;
		foreach aWeapName(WName) {
			if (nl) InfoBuf $= ",";
			InfoBuf $= WName;
			nl = true;
		}
	}
	
	//!hawawa: �X�y�V�����ȉ�������������E�E�E�����H
	function Broadcast_Special(string Msg) {
		local bool nonemeg;
		nonemeg = (Msg=="");
		switch (Msg) {
			case "=o":
				SendRestrictMessageStringAll("::�ӂɂႠ�[�[���H�I");
				break;
			case ":o":
				SendRestrictMessageStringAll("::�͂���A�т����肵���̂ł��I");
				break;
			default:
				if (nonemeg) {
					SendRestrictMessageStringAll("::�͂ɂႠ�[���H�I");
				}else{
					SendRestrictMessageStringAll("::����ȑ����ő��v�Ȃ̂ł��H");
				}
				break;
		}
	}

/*

	case "!dappun":
		if (MsgBody=="") Broadcast_Puke(GetKFPCFromPRI(SenderPRI));
		break;		
	
	//!dappun: ���V�� ����2
	function Broadcast_Puke(KFPlayerController KFPC) {
		local KFPawn_Human KFPH;
		local byte i,MineNum;
		local float ExplodeTimer;
		local rotator DPMR; //DeathPukeMineRotations_Mine
		local KFProjectile PukeMine;
		KFPH = KFPawn_Human(KFPC.Pawn);
		MineNum = 5;
		ExplodeTimer = 5.0f;
		for (i=0;i<MineNum;++i) {
			DPMR.Pitch = 8190;
			DPMR.Yaw = (2*32768)*i/MineNum-32768;
			DPMR.Roll = 0;
			PukeMine = Spawn(class'RPWPukeMine', KFPH,, KFPH.Location, DPMR,, true);
			if( PukeMine != none ) {
				PukeMine.Init( vector(DPMR) );
			}
			RPWPukeMine(PukeMine).SetExplodeTimer(ExplodeTimer);
		}
		SendRestrictMessageStringAll("::���܂݂�ɂȂ낤��B");
	}
*/
	
/////////////////////////////////////////<<---EOF--->>/////////////////////////////////////////