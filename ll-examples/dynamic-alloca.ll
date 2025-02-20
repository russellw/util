; https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:___c,selection:(endColumn:2,endLineNumber:33,positionColumn:2,positionLineNumber:33,selectionStartColumn:2,selectionStartLineNumber:33,startColumn:2,startLineNumber:33),source:'%23include+%3Cstdio.h%3E%0A%23include+%3Cstdlib.h%3E%0A%23include+%3Calloca.h%3E%0A%0A//+This+function+forces+dynamic+alloca+by+using+a+variable+size%0Avoid+process_dynamic_array(int+size)+%7B%0A++++//+alloca+allocates+!'size!'+bytes+on+the+stack%0A++++int+*array+%3D+(int+*)alloca(size+*+sizeof(int))%3B%0A++++%0A++++//+Initialize+the+array+with+some+values%0A++++for+(int+i+%3D+0%3B+i+%3C+size%3B+i%2B%2B)+%7B%0A++++++++array%5Bi%5D+%3D+i+*+2%3B%0A++++%7D%0A++++%0A++++//+Use+the+array+to+force+the+compiler+to+actually+allocate+it%0A++++int+sum+%3D+0%3B%0A++++for+(int+i+%3D+0%3B+i+%3C+size%3B+i%2B%2B)+%7B%0A++++++++sum+%2B%3D+array%5Bi%5D%3B%0A++++%7D%0A++++%0A++++printf(%22Sum+of+array+elements:+%25d%5Cn%22,+sum)%3B%0A%7D%0A%0Aint+main(int+argc,+char+*argv%5B%5D)+%7B%0A++++//+Get+size+from+command+line+or+use+default%0A++++int+size+%3D+(argc+%3E+1)+%3F+atoi(argv%5B1%5D)+:+5%3B%0A++++%0A++++//+This+will+force+clang+to+use+dynamic+alloca%0A++++//+since+the+size+isn!'t+known+at+compile+time%0A++++process_dynamic_array(size)%3B%0A++++%0A++++return+0%3B%0A%7D'),l:'5',n:'0',o:'C+source+%231',t:'0')),k:33.333333333333336,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:cclang1910,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'0',fontScale:14,fontUsePx:'0',j:1,lang:___c,libs:!(),options:'-emit-llvm+-O3',overrides:!(),selection:(endColumn:29,endLineNumber:147,positionColumn:29,positionLineNumber:147,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+clang+19.1.0+(Editor+%231)',t:'0')),k:33.333333333333336,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:flags,i:(compilerFlags:'-emit-llvm+-O3',compilerName:'x86-64+clang+19.1.0',editorid:1,fontScale:14,fontUsePx:'0',j:1,treeid:0),l:'5',n:'0',o:'Detailed+Compiler+Flags+x86-64+clang+19.1.0+(Editor+%231,+Compiler+%231)',t:'0')),k:33.33333333333333,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
@.str = private unnamed_addr constant [27 x i8] c"Sum of array elements: %d\0A\00", align 1, !dbg !0

define dso_local void @process_dynamic_array(i32 noundef %0) local_unnamed_addr #0 !dbg !24 {
    #dbg_value(i32 %0, !28, !DIExpression(), !35)
  %2 = sext i32 %0 to i64, !dbg !36
  %3 = shl nsw i64 %2, 2, !dbg !36
  %4 = alloca i8, i64 %3, align 16, !dbg !36
    #dbg_value(ptr %4, !29, !DIExpression(), !35)
    #dbg_value(i32 0, !30, !DIExpression(), !37)
  %5 = icmp sgt i32 %0, 0, !dbg !38
  br i1 %5, label %6, label %58, !dbg !40

6:
  %7 = zext nneg i32 %0 to i64, !dbg !38
  %8 = icmp ult i32 %0, 8, !dbg !40
  br i1 %8, label %9, label %11, !dbg !40

9:
  %10 = phi i64 [ 0, %6 ], [ %12, %24 ]
  br label %51, !dbg !40

11:
  %12 = and i64 %7, 2147483640, !dbg !40
  br label %13, !dbg !40

13:
  %14 = phi i64 [ 0, %11 ], [ %21, %13 ], !dbg !41
  %15 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %11 ], [ %22, %13 ], !dbg !42
  %16 = getelementptr inbounds i32, ptr %4, i64 %14, !dbg !44
  %17 = shl <4 x i32> %15, <i32 1, i32 1, i32 1, i32 1>, !dbg !42
  %18 = shl <4 x i32> %15, <i32 1, i32 1, i32 1, i32 1>, !dbg !42
  %19 = add <4 x i32> %18, <i32 8, i32 8, i32 8, i32 8>, !dbg !42
  %20 = getelementptr inbounds i8, ptr %16, i64 16, !dbg !42
  store <4 x i32> %17, ptr %16, align 16, !dbg !42
  store <4 x i32> %19, ptr %20, align 16, !dbg !42
  %21 = add nuw i64 %14, 8, !dbg !41
  %22 = add <4 x i32> %15, <i32 8, i32 8, i32 8, i32 8>, !dbg !42
  %23 = icmp eq i64 %21, %12, !dbg !41
  br i1 %23, label %24, label %13, !dbg !41

24:
  %25 = icmp eq i64 %12, %7, !dbg !40
  br i1 %25, label %26, label %9, !dbg !40

26:
    #dbg_value(i32 0, !33, !DIExpression(), !54)
    #dbg_value(i32 0, !32, !DIExpression(), !35)
  br i1 %5, label %27, label %58, !dbg !55

27:
  %28 = zext nneg i32 %0 to i64, !dbg !56
  %29 = icmp ult i32 %0, 8, !dbg !55
  br i1 %29, label %30, label %33, !dbg !55

30:
  %31 = phi i64 [ 0, %27 ], [ %34, %47 ]
  %32 = phi i32 [ 0, %27 ], [ %49, %47 ]
  br label %61, !dbg !55

33:
  %34 = and i64 %7, 2147483640, !dbg !55
  br label %35, !dbg !55

35:
  %36 = phi i64 [ 0, %33 ], [ %45, %35 ], !dbg !58
  %37 = phi <4 x i32> [ zeroinitializer, %33 ], [ %43, %35 ]
  %38 = phi <4 x i32> [ zeroinitializer, %33 ], [ %44, %35 ]
  %39 = getelementptr inbounds i32, ptr %4, i64 %36, !dbg !59
  %40 = getelementptr inbounds i8, ptr %39, i64 16, !dbg !59
  %41 = load <4 x i32>, ptr %39, align 16, !dbg !59
  %42 = load <4 x i32>, ptr %40, align 16, !dbg !59
  %43 = add <4 x i32> %41, %37, !dbg !61
  %44 = add <4 x i32> %42, %38, !dbg !61
  %45 = add nuw i64 %36, 8, !dbg !58
  %46 = icmp eq i64 %45, %34, !dbg !58
  br i1 %46, label %47, label %35, !dbg !58

47:
  %48 = add <4 x i32> %44, %43, !dbg !55
  %49 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %48), !dbg !55
  %50 = icmp eq i64 %34, %7, !dbg !55
  br i1 %50, label %58, label %30, !dbg !55

51:
  %52 = phi i64 [ %56, %51 ], [ %10, %9 ]
    #dbg_value(i64 %52, !30, !DIExpression(), !37)
  %53 = getelementptr inbounds i32, ptr %4, i64 %52, !dbg !44
  %54 = trunc i64 %52 to i32, !dbg !42
  %55 = shl i32 %54, 1, !dbg !42
  store i32 %55, ptr %53, align 4, !dbg !42
  %56 = add nuw nsw i64 %52, 1, !dbg !41
    #dbg_value(i64 %56, !30, !DIExpression(), !37)
  %57 = icmp eq i64 %56, %7, !dbg !38
  br i1 %57, label %26, label %51, !dbg !40

58:
  %59 = phi i32 [ 0, %26 ], [ 0, %1 ], [ %49, %47 ], [ %66, %61 ], !dbg !35
  %60 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %59), !dbg !65
  ret void, !dbg !66

61:
  %62 = phi i64 [ %67, %61 ], [ %31, %30 ]
  %63 = phi i32 [ %66, %61 ], [ %32, %30 ]
    #dbg_value(i64 %62, !33, !DIExpression(), !54)
    #dbg_value(i32 %63, !32, !DIExpression(), !35)
  %64 = getelementptr inbounds i32, ptr %4, i64 %62, !dbg !59
  %65 = load i32, ptr %64, align 4, !dbg !59
  %66 = add nsw i32 %65, %63, !dbg !61
    #dbg_value(i32 %66, !32, !DIExpression(), !35)
  %67 = add nuw nsw i64 %62, 1, !dbg !58
    #dbg_value(i64 %67, !33, !DIExpression(), !54)
  %68 = icmp eq i64 %67, %28, !dbg !56
  br i1 %68, label %58, label %61, !dbg !55
}

declare !dbg !68 noundef i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

define dso_local noundef i32 @main(i32 noundef %0, ptr nocapture noundef readonly %1) local_unnamed_addr #0 !dbg !75 {
    #dbg_value(i32 %0, !79, !DIExpression(), !82)
    #dbg_value(ptr %1, !80, !DIExpression(), !82)
  %3 = icmp sgt i32 %0, 1, !dbg !83
  br i1 %3, label %4, label %9, !dbg !84

4:
  %5 = getelementptr inbounds i8, ptr %1, i64 8, !dbg !85
  %6 = load ptr, ptr %5, align 8, !dbg !85
    #dbg_value(ptr %6, !88, !DIExpression(), !94)
  %7 = tail call i64 @strtol(ptr nocapture noundef nonnull %6, ptr noundef null, i32 noundef 10) #4, !dbg !96
  %8 = trunc i64 %7 to i32, !dbg !97
  br label %9, !dbg !84

9:
  %10 = phi i32 [ %8, %4 ], [ 5, %2 ], !dbg !84
    #dbg_value(i32 %10, !81, !DIExpression(), !82)
  tail call void @process_dynamic_array(i32 noundef %10), !dbg !98
  ret i32 0, !dbg !99
}

declare !dbg !100 i64 @strtol(ptr noundef readonly, ptr nocapture noundef, i32 noundef) local_unnamed_addr #2

declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #3

attributes #0 = { nofree nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree nounwind willreturn "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }