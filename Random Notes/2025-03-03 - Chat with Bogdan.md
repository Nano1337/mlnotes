
We talked about scaling our current pipeline systems

Death by a thousand cuts is the usual way he explains how while things work at a smaller scale, they might not necessarily work when you scale them all the way up. A lot of tiny issues will cause bottlenecks and cause really big issues at a larger scale. 

I'm currently running into these issues when trying to scale the templated recaptioning job, and I really do feel this is the case.

And especially with the distributed systems and systems at scale, I feel like it's the minor improvements that add up together to create something great and super efficient. So you just have to persevere and keep working at it. 

We also talked a little bit about the data loader. Their DeepSeek 3FS system is really cool. And the WEKA system is a startup that does basically that with RDMA direct access, instead of having to go through the Linux kernel, completely bypasses it and does the random access directly, which is much faster with the direct and with the DMA. Now, on this side, with the S3 system constraint, we might have to... Well, we can do this RDMA stuff directly with FSx. We can do some hacky stuff there. Apparently, from what I remember... Apparently, Daft currently has a data loader where you can maximally pack the S3 API read call and kinda squeeze as much throughput through the S3. So that might be looking maybe looking useful to look into. Also they orchestrate everything with the ray. So maybe if we can build a ray data loader that's super efficient and uses the S3 API correctly, we're going to orchestrate something really cool and distribute it at scale. 

The only point of problem is that if Cohere or someone is running a large training job at scale. If this data loader distributed system is not reliable enough and if there's any downtime, then that would result in a lot of people being unhappy and would come to you and call you. So if you want to take this route, you have to be very careful and really have to ensure reliability. 
