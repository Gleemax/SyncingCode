class PowerZeds_KFP_Fleshpound extends KFPawn_ZedFleshPound;

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
	//GoreHealth was 650
    HitZones[HZI_HEAD]=(ZoneName=head, BoneName=Head, Limb=BP_Head, GoreHealth=433, DmgScale=1.1, SkinID=1)
}