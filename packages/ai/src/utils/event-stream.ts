import type { AssistantMessage, AssistantMessageEvent } from "../types.ts";

class FifoQueue<T> {
	private incoming: T[] = [];
	private outgoing: T[] = [];

	get length(): number {
		return this.incoming.length + this.outgoing.length;
	}

	enqueue(value: T): void {
		this.incoming.push(value);
	}

	dequeue(): T | undefined {
		if (this.outgoing.length === 0) {
			while (this.incoming.length > 0) {
				this.outgoing.push(this.incoming.pop()!);
			}
		}
		return this.outgoing.pop();
	}
}

// Generic event stream class for async iteration
/**
 * EN: FIFO bridge from producer callbacks to async iteration, with a separate final-result promise. A
 * terminal event resolves result(), but queued events remain available to the iterator.
 *
 * ZH: 把生产者回调接到异步迭代的 FIFO 桥，同时提供独立的最终结果 Promise。终止事件会完成 result()，但队列中的事件仍可继续由迭代器读取。
 *
 * EN: This is a consuming queue, not a broadcast or backpressure mechanism. Consumers share queued events,
 * and push does not wait for their work. The producer must provide a terminal event or a final result to
 * settle result().
 *
 * ZH: 这是消费队列，不是广播或背压机制。多个消费者共享队列事件，push 不等待消费工作。生产者必须提供终止事件或最终结果，才能让 result() 完成。
 */
export class EventStream<T, R = T> implements AsyncIterable<T> {
	private queue = new FifoQueue<T>();
	private waiting = new FifoQueue<(value: IteratorResult<T>) => void>();
	private done = false;
	private finalResultPromise: Promise<R>;
	private resolveFinalResult!: (result: R) => void;
	private isComplete: (event: T) => boolean;
	private extractResult: (event: T) => R;

	constructor(isComplete: (event: T) => boolean, extractResult: (event: T) => R) {
		this.isComplete = isComplete;
		this.extractResult = extractResult;
		this.finalResultPromise = new Promise((resolve) => {
			this.resolveFinalResult = resolve;
		});
	}

	/**
	 * EN: Ignore events after termination. Resolve the final result on a terminal event, then deliver that
	 * event to one waiter or enqueue it. Terminal-event delivery and iterator completion are distinct steps.
	 *
	 * ZH: 终止后忽略新事件。遇到终止事件时先完成最终结果，再把该事件交给一个等待者或加入队列。终止事件的投递与迭代器结束是两个步骤。
	 */
	push(event: T): void {
		if (this.done) return;

		if (this.isComplete(event)) {
			this.done = true;
			this.resolveFinalResult(this.extractResult(event));
		}

		// Deliver to waiting consumer or queue it
		const waiter = this.waiting.dequeue();
		if (waiter) {
			waiter({ value: event, done: false });
		} else {
			this.queue.enqueue(event);
		}
	}

	/**
	 * EN: Close iteration and release all pending iterator waiters. Only a supplied result resolves the result
	 * promise here; closing without a result or prior terminal event leaves that promise pending.
	 *
	 * ZH: 关闭迭代并释放所有等待中的迭代器。此处只有传入 result 才会完成结果 Promise；若没有 result 且此前没有终止事件，该 Promise 会继续等待。
	 */
	end(result?: R): void {
		this.done = true;
		if (result !== undefined) {
			this.resolveFinalResult(result);
		}
		// Notify all waiting consumers that we're done
		while (this.waiting.length > 0) {
			const waiter = this.waiting.dequeue()!;
			waiter({ value: undefined as any, done: true });
		}
	}

	async *[Symbol.asyncIterator](): AsyncIterator<T> {
		while (true) {
			if (this.queue.length > 0) {
				yield this.queue.dequeue()!;
			} else if (this.done) {
				return;
			} else {
				const result = await new Promise<IteratorResult<T>>((resolve) => this.waiting.enqueue(resolve));
				if (result.done) return;
				yield result.value;
			}
		}
	}

	/**
	 * EN: Wait for the terminal result independently of event consumption. This does not drain the event queue
	 * or wait for UI subscribers to finish processing events.
	 *
	 * ZH: 独立于事件消费等待最终结果。它不会排空事件队列，也不会等待 UI 订阅者处理完事件。
	 */
	result(): Promise<R> {
		return this.finalResultPromise;
	}
}

/**
 * EN: Specialize the generic stream for assistant responses. Both done and error resolve the final
 * AssistantMessage; consumers inspect stopReason rather than expecting error events to reject result().
 *
 * ZH: 将通用流特化为 assistant 响应流。done 与 error 都会完成最终 AssistantMessage；消费者需检查 stopReason，而不是期待 error 事件让 result()
 * 拒绝。
 */
export class AssistantMessageEventStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") {
					return event.message;
				} else if (event.type === "error") {
					return event.error;
				}
				throw new Error("Unexpected event type for final result");
			},
		);
	}
}

/** Factory function for AssistantMessageEventStream (for use in extensions) */
export function createAssistantMessageEventStream(): AssistantMessageEventStream {
	return new AssistantMessageEventStream();
}
