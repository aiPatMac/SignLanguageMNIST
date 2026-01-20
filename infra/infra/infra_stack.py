from aws_cdk import (
    Stack,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_lambda_event_sources as lambda_events,
    RemovalPolicy
)
from constructs import Construct

class InfraStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(self, "MnistInferenceBucket",
            removal_policy=RemovalPolicy.DESTROY, 
            auto_delete_objects=True             
        )

        mnist_lambda = _lambda.DockerImageFunction(
            self, "SignMnistLambda",
            code=_lambda.DockerImageCode.from_image_asset("../", file="Dockerfile.lambda"),
            memory_size=2048, 
            timeout=Duration.seconds(60)
        )

        bucket.grant_read_write(mnist_lambda)

        mnist_lambda.add_event_source(
            lambda_events.S3EventSource(bucket,
                events=[s3.EventType.OBJECT_CREATED],
                filters=[s3.NotificationKeyFilter(suffix=".jpg")]
            )
        )