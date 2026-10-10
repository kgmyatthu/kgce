#include "FXConfig.h"

#include "Lighting.fx_fragment"
#include "ShadowMap.fx_fragment"
#include "Util.fx_fragment"
#include "ScreenEffects.fx_fragment"
#include "standard_rigid_matrices.fx_fragment"
#include "Camera.fx_fragment"
#include "fog.fx_fragment"
#include "Blending.fx_fragment"
#include "frameview_mrt.fx_fragment"
#include "Shroud.fx_fragment"

struct VS_INPUT_TEXTURED
{	
	float3	position	: POSITION;
	float4	normal		: NORMAL;
	float2	texture_uv	: TEXCOORD0;
};

struct VS_INPUT_TEXTURED_INSTANCED
{	
	float3	position		: POSITION;	
	float4	normal			: NORMAL;
	float2	texture_uv		: TEXCOORD0;
	
	float4	instance_index	: BLENDINDICES;
};

struct VS_INPUT_TEXTURED_INSTANCED_30
{	
	float3	position		: POSITION;		
	float4	normal			: NORMAL;
	float2	texture_uv		: TEXCOORD0;
	
	float4	matrix_row1		: POSITION1;
	float4	matrix_row2		: POSITION2;
	float4	matrix_row3		: POSITION3;
	float4	matrix_row4		: POSITION4;
};

struct VS_OUTPUT_TEXTURED
{
	float4	position						: POSITION;
	float4	world_position					: TEXCOORD0;
	float2	texture_uv						: TEXCOORD1;
	half4	shadow							: TEXCOORD2;
	float3	normal							: TEXCOORD3;
	float3	diffuse_color					: COLOR0;
	float3	ambient_color					: COLOR1;
};

DECLARE_SRGB_TEXTURE(diffuse_map)

#define RIGID_MESH_FX

#define NO_INSTANCING
#include "campaignbuilding_vertex.fx_fragment"
#undef  NO_INSTANCING

#define INSTANCING
#include "campaignbuilding_vertex.fx_fragment"
#undef  INSTANCING

#define INSTANCING_30
#include "campaignbuilding_vertex.fx_fragment"
#undef  INSTANCING_30

#undef RIGID_MESH_FX
//---------------------------------------------------------------------------------------------
float4 pixel_textured(const VS_OUTPUT_TEXTURED v, uniform bool sm2=false) : COLOR
{
	half occlusion			= CALC_OCCLUSION_SMCHECK(v, !sm2);
	occlusion				*= shroud(v.world_position);

	float4 colour	= tex2D(s_diffuse_map, v.texture_uv);
	
	colour.rgb *= v.diffuse_color * occlusion + v.ambient_color;
	colour.rgb *= visible(v.world_position);
	
	colour = campaign_fog(colour, v.world_position.xyz, sm2);
	return hdr_encode(colour);			
}

float4 shadow_map_pixel_with_alpha_map(const VS_SHADOW_MAP_WITH_ALPHA_MAP_OUTPUT v,uniform bool ps_30) : COLOR
{
	float4 diffuse_pixel = tex2D(s_diffuse_map, v.tex_uv);
	clip(diffuse_pixel.a-0.01f);
	float scale_dist=v.distance.x/v.distance.w;	
	if (ps_30)
	{
		scale_dist += get_shadow_slope_bias(scale_dist);
	}
	return float4(scale_dist,scale_dist,scale_dist,1);
}


#define DEPTH_WRITE_RENDER_STATES \
		ZENABLE = TRUE; \
		ZWRITEENABLE = TRUE; \
		COLORWRITEENABLE	= DEPTH_PASS_COLORWRITEENABLE;		
		
#if defined(DEPTH_PRE_PASS)		
#define DEPTH_TEST_RENDER_STATES \
 		ZWRITEENABLE		= FALSE; \
		ZENABLE				= TRUE; \
		ZFUNC				= EQUAL; 
#define PRE_PASS true
#else
#define DEPTH_TEST_RENDER_STATES \
 		ZWRITEENABLE		= TRUE; \
		ZENABLE				= TRUE; 
#define PRE_PASS false
#endif	

technique campaign_building_instanced_vs30_sm3
{
	pass p0
    {
		VertexShader = compile vs_3_0 textured_vertex_instanced_30();
#ifdef LOW_QUALITY_SHADERS
		PixelShader  = compile ps_3_0 pixel_textured(true);		
#else
		PixelShader  = compile ps_3_0 pixel_textured();		
#endif //LOW_QUALITY_SHADERS
		
		DEPTH_TEST_RENDER_STATES				
	}
}

technique campaign_building_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 textured_vertex();
		PixelShader  = compile ps_2_0 pixel_textured(true);				
		
		DEPTH_TEST_RENDER_STATES				
	}
}

technique campaign_building_instanced_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 textured_vertex_instanced();
		PixelShader	 = compile ps_2_0 pixel_textured(true);
		
		DEPTH_TEST_RENDER_STATES				
	}
}

/********************************************************************************************/
technique depth_campaign_building_instanced_vs30_sm3
{
	pass p0
    {
		VertexShader = compile vs_3_0 depth_textured_vertex_instanced_30();
		PixelShader	 = compile ps_3_0 depth_buffer_pixel_alpha_texture2(s_diffuse_map);		

		DEPTH_WRITE_RENDER_STATES
	}
}

technique depth_campaign_building_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 depth_textured_vertex();
		PixelShader	 = compile ps_2_0 depth_buffer_pixel_alpha_texture2(s_diffuse_map);		

		DEPTH_WRITE_RENDER_STATES
	}
}

technique depth_campaign_building_instanced_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 depth_textured_vertex_instanced();
		PixelShader	 = compile ps_2_0 depth_buffer_pixel_alpha_texture2(s_diffuse_map);		

		DEPTH_WRITE_RENDER_STATES
	}
}
/*******************************************************************************************/
technique shadow_campaign_building_instanced_vs30_sm3
{
	pass p0
    {
		VertexShader = compile vs_3_0 shadow_textured_vertex_instanced_30();
		PixelShader	 = compile ps_3_0 shadow_map_pixel_with_alpha_map(true);		
		DECLARE_SHADOW_SHADER_CONFIG
	}
}

technique shadow_campaign_building_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 shadow_textured_vertex();
		PixelShader	 = compile ps_2_0 shadow_map_pixel_with_alpha_map(false);		
		DECLARE_SHADOW_SHADER_CONFIG
	}
}

technique shadow_campaign_building_instanced_sm2
{
	pass p0
    {
		VertexShader = compile vs_2_0 shadow_textured_vertex_instanced();
		PixelShader	 = compile ps_2_0 shadow_map_pixel_with_alpha_map(false);		
		DECLARE_SHADOW_SHADER_CONFIG
	}
}


