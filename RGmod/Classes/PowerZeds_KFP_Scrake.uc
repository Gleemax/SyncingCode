class PowerZeds_KFP_Scrake extends KFPawn_ZedScrake;
//class PowerZeds_KFP_Scrake extends KFPawn_ZedScrake_Versus;
	
//見た目はVS
simulated event bool UsePlayerControlledZedSkin() {
	return true;
}

//おう常に怒るのやめろォ！
simulated event Tick( float DeltaTime ){
	super.Tick( DeltaTime );
	if (!IsEnraged()) SetEnraged(true);
}

DefaultProperties
{
	//GoreHealth was 600
    HitZones[HZI_HEAD]=(ZoneName=head, BoneName=Head, Limb=BP_Head, GoreHealth=400, DmgScale=1.1, SkinID=1)
}