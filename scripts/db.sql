create table if not exists Scores
(	MSISDN_Number float not null,
    Engagement_Score_Dur float not null, 
    Engagement_Score_Total float not null,
    Engagement_Satisf_Score float not null, 
    Experience_Score_Total_TCP float not null,
    Experience_Score_Total_RTT float not null,
    Experience_Score_Total_TP float not null,
    Experience_Satisf_Score float not null,
    PRIMARY KEY(MSISDN_Number)
)
ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;